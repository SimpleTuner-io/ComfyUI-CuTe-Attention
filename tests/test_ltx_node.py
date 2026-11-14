"""CPU integration contracts; CUDA arithmetic is tested separately."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from cute_attention import ltx_node
from cute_attention.vendor.kohakufa_cute import neighborhood_attention_3d

comfy_kitchen = SimpleNamespace(na3d=lambda q, k, v, kernel, causal, scale: q, marker=object())


class Block(torch.nn.Module):
    def forward(self, q, pre=None):
        if pre:
            q = pre(q)
        return comfy_kitchen.na3d(q, q, q, [3, 7, 7], None, 1.0)


class LTXNodeTests(unittest.TestCase):
    def test_adapter_layout_clipping_scale_and_strides(self):
        q = torch.randn(1, 2, 3, 4, 2, 64).transpose(2, 3)
        with patch(
            "cute_attention.vendor.kohakufa_cute.api.neighborhood_attention",
            return_value=q.flatten(1, 3).transpose(1, 2),
        ) as call:
            out = neighborhood_attention_3d(q, q, q, (11, 3, 7), None, 1.0)
        torch.testing.assert_close(out, q)
        self.assertEqual(
            call.call_args.kwargs, {"spatial_shape": (2, 4, 3), "window_size": (2, 3, 3), "scale": 1.0}
        )
        torch.testing.assert_close(call.call_args.args[0], q.flatten(1, 3).transpose(1, 2))

    def test_adapter_rejects_unsupported_contracts(self):
        q = torch.zeros(1, 2, 3, 4, 1, 64)
        for causal in (True, [False, True, False], [False, False]):
            with self.assertRaises(ValueError):
                neighborhood_attention_3d(q, q, q, 3, causal)
        for kernel in (0, [3, 3], [3, 0, 3]):
            with self.assertRaises(ValueError):
                neighborhood_attention_3d(q, q, q, kernel)
        with self.assertRaises(ValueError):
            neighborhood_attention_3d(q, q, q[..., :32], 3)
        with self.assertRaises(TypeError):
            neighborhood_attention_3d(q, q, q, 3, attn_mask=torch.ones(1))

    def test_scope_thread_isolation_and_exception_restore(self):
        model = torch.nn.Sequential(Block(), Block())
        q = torch.zeros(1, 2, 3, 4, 1, 64)
        original = [b.forward for b in model]
        with patch.object(ltx_node, "neighborhood_attention_3d", side_effect=lambda q, *a: q + 1):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with ltx_node.ltx_neighborhood(model, Block) as backend:
                    torch.testing.assert_close(model(q), q + 2)
                    self.assertIs(backend.marker, comfy_kitchen.marker)
                    self.assertEqual(backend.calls, 2)
                    with ThreadPoolExecutor(1) as pool:
                        torch.testing.assert_close(pool.submit(model, q).result(), q)
                    raise RuntimeError("deliberate")
        self.assertEqual([b.forward for b in model], original)
        self.assertTrue(all("forward" not in b.__dict__ for b in model))

    def test_partial_setup_and_existing_override(self):
        model = torch.nn.Sequential(Block(), Block())
        override = lambda q: q
        model[1].forward = override
        with self.assertRaises(RuntimeError):
            with ltx_node.ltx_neighborhood(model, Block):
                pass
        self.assertNotIn("forward", model[0].__dict__)
        self.assertIs(model[1].forward, override)
        with self.assertRaises(ValueError):
            with ltx_node.ltx_neighborhood(torch.nn.Identity(), Block):
                pass
        model[0]._compiled_call_impl = lambda q: q
        with self.assertRaisesRegex(RuntimeError, "Compiled"):
            with ltx_node.ltx_neighborhood(model, Block):
                pass

    def test_auto_cpu_fallback_and_strict_no_conversion(self):
        q = torch.zeros(1, 2, 3, 4, 1, 64)
        backend = ltx_node.LTXNeighborhoodBackend(comfy_kitchen, "auto")
        self.assertIs(backend.na3d(q, q, q, [3, 7, 7], None, 1.0), q)
        self.assertEqual((backend.calls, backend.fallback_calls), (0, 1))
        backend = ltx_node.LTXNeighborhoodBackend(comfy_kitchen, "cute")
        with self.assertRaises(ValueError):
            backend.na3d(q, q, q, [3, 7, 7], None, 1.0)
        self.assertEqual(backend.fallback_calls, 0)

    def test_node_delegates_regular_and_tiled_decode(self):
        model = torch.nn.Sequential(Block())
        vae, samples = SimpleNamespace(first_stage_model=model), {"samples": object()}
        regular, tiled = Mock(), Mock()
        q = torch.zeros(1, 2, 3, 4, 1, 64)
        regular.return_value.decode.side_effect = lambda *a: (model(q),)
        tiled.return_value.decode.side_effect = lambda *a: (model(q),)
        modules = {
            "nodes": SimpleNamespace(VAEDecode=regular, VAEDecodeTiled=tiled),
            "comfy.ldm.lightricks.vae.na_diffusion_decoder": SimpleNamespace(NeighborhoodAttention3D=Block),
        }
        with patch.dict("sys.modules", modules):
            node = ltx_node.CuTeLTXVAE()
            torch.testing.assert_close(node.decode(samples, vae, "comfy")[0], q)
            node.decode(samples, vae, "comfy", True, 256, 32, 24, 8)
        regular.return_value.decode.assert_called_once_with(vae, samples)
        tiled.return_value.decode.assert_called_once_with(vae, samples, 256, 32, 24, 8)
        self.assertNotIn("forward", model[0].__dict__)

    def test_empty_decode_is_not_reported_as_cute_execution(self):
        vae = SimpleNamespace(first_stage_model=torch.nn.Sequential(Block()))
        modules = {
            "nodes": SimpleNamespace(VAEDecode=lambda: SimpleNamespace(decode=lambda *a: (None,))),
            "comfy.ldm.lightricks.vae.na_diffusion_decoder": SimpleNamespace(NeighborhoodAttention3D=Block),
        }
        with patch.dict("sys.modules", modules), self.assertRaisesRegex(RuntimeError, "no supported"):
            ltx_node.CuTeLTXVAE().decode({}, vae)


if __name__ == "__main__":
    unittest.main()
