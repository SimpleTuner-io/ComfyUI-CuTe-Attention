"""Real-kernel validation: python -m tests.gpu_validation [--comfyui-root PATH]."""

import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from cute_attention import runtime
from cute_attention.attention import ComfyAttention
from cute_attention.vendor.kohakufa_cute import api

from .reference import fp64_reference

COMFYUI_ROOT = None


class GPUValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or not api._runtime_available():
            raise RuntimeError("A supported CUDA GPU and CuTe runtime dependencies are required.")
        torch.manual_seed(1234)
        torch.set_num_threads(4)
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "capability": torch.cuda.get_device_capability(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                }
            ),
            flush=True,
        )

    def tearDown(self):
        runtime.set_process_mode(None)

    def test_real_kernels_against_fp64_outputs_and_gradients(self):
        cases = (
            (64, False, False, False),
            (64, False, True, True),
            (128, True, False, False),
            (256, False, False, False),
            (512, False, False, False),
        )
        for dtype in (torch.float16, torch.bfloat16):
            for dim, causal, masked, gqa in cases:
                with self.subTest(dtype=dtype, dim=dim, causal=causal, masked=masked, gqa=gqa):
                    q = (torch.randn(2, 65, 4, dim, device="cuda", dtype=dtype) * 0.5).transpose(1, 2)
                    k = (
                        torch.randn(2, 79, 2 if gqa else 4, dim, device="cuda", dtype=dtype) * 0.5
                    ).transpose(1, 2)
                    v = torch.randn_like(k)
                    inputs = tuple(t.requires_grad_() for t in (q, k, v))
                    reference_inputs = tuple(t.detach().double().requires_grad_() for t in inputs)
                    mask = None
                    if masked:
                        mask = torch.rand(65, 79, device="cuda") > 0.3
                        mask[1] = False
                    options = dict(attn_mask=mask, is_causal=causal, enable_gqa=gqa, scale=-0.125)
                    expected = runtime._native_sdpa(*reference_inputs, **options)
                    actual = api.scaled_dot_product_attention(*inputs, **options)
                    tolerance = 0.012 if dtype == torch.bfloat16 else 0.003
                    torch.testing.assert_close(actual.double(), expected, rtol=tolerance, atol=tolerance)
                    if masked:
                        self.assertEqual(actual[:, :, 1].count_nonzero().item(), 0)
                    dout = torch.randn_like(actual)
                    gradients = torch.autograd.grad(actual, inputs, dout)
                    reference_gradients = torch.autograd.grad(expected, reference_inputs, dout.double())
                    errors = []
                    for got, wanted in zip(gradients, reference_gradients):
                        error = ((got.double() - wanted).norm() / wanted.norm().clamp_min(1e-12)).item()
                        self.assertLess(error, 0.025 if dtype == torch.bfloat16 else 0.006)
                        errors.append(error)
                    print(
                        json.dumps(
                            {
                                "dtype": str(dtype),
                                "dim": dim,
                                "causal": causal,
                                "masked": masked,
                                "gqa": gqa,
                                "gradient_relative_errors": errors,
                            }
                        ),
                        flush=True,
                    )
        self.assertEqual(api._extension.cache_info().currsize, 1)

    def test_process_scope_fallback_after_native_extension_is_warm(self):
        q = torch.randn(1, 4, 17, 64, device="cuda", dtype=torch.bfloat16)
        api.scaled_dot_product_attention(q, q, q)
        runtime.set_process_mode("auto")
        for mask in (torch.randn(17, 17, device="cuda", dtype=q.dtype), None):
            options = dict(attn_mask=mask, scale=0.25)
            actual = F.scaled_dot_product_attention(q, q, q, **options)
            expected = runtime._native_sdpa(q, q, q, **options)
            if mask is None:
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.015)
            else:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        fp32 = q.float()
        torch.testing.assert_close(
            F.scaled_dot_product_attention(fp32, fp32, fp32),
            runtime._native_sdpa(fp32, fp32, fp32),
            rtol=0,
            atol=0,
        )
        runtime.set_process_mode(None)
        torch.testing.assert_close(
            F.scaled_dot_product_attention(q, q, q), runtime._native_sdpa(q, q, q), rtol=0, atol=0
        )

    def test_comfy_layouts_reach_strict_cute(self):
        q = torch.randn(2, 65, 4, 64, device="cuda", dtype=torch.float16).transpose(1, 2)
        k = torch.randn(2, 79, 2, 64, device="cuda", dtype=q.dtype).transpose(1, 2)
        v = torch.randn_like(k)
        mask = torch.rand(2, 65, 79, device="cuda") > 0.3
        expected = runtime._native_sdpa(
            q.double(), k.double(), v.double(), mask.unsqueeze(1), enable_gqa=True
        )
        for preshaped in (True, False):
            for head_output in (True, False):
                inputs = (q, k, v)
                if not preshaped:
                    inputs = tuple(t.transpose(1, 2).reshape(2, t.shape[2], -1) for t in inputs)
                out = ComfyAttention("strict")(
                    *inputs,
                    4,
                    mask=mask,
                    skip_reshape=preshaped,
                    skip_output_reshape=head_output,
                    enable_gqa=True,
                )
                wanted = expected if head_output else expected.transpose(1, 2).reshape(2, 65, -1)
                torch.testing.assert_close(out.double(), wanted, rtol=0.004, atol=0.004)

    def test_vendored_compiler_path(self):
        q = torch.randn(1, 4, 17, 64, device="cuda", dtype=torch.float16)
        compiled = torch.compile(api.automatic_scaled_dot_product_attention, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(q, q, q), api.scaled_dot_product_attention(q, q, q))

    def assert_centered_reference(self, actual, inputs, dout, causal, scale=None):
        expected = fp64_reference(inputs, dout, causal=causal, scale=scale)
        errors = []
        for got, wanted in zip(actual, expected):
            self.assertTrue(torch.isfinite(got).all().item())
            denominator = wanted.norm().item()
            if denominator:
                error = (got.double() - wanted).norm().item() / denominator
                self.assertLess(error, 0.01 if inputs[0].dtype == torch.bfloat16 else 0.005)
                errors.append(error)
            else:
                self.assertEqual(got.count_nonzero().item(), 0)
                errors.append(0.0)
        return errors

    def test_rtx4090_schedule_boundaries(self):
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090":
            self.skipTest("Requires the full RTX 4090 dispatch predicate.")
        from cute_attention.vendor.kohakufa_cute import ada, kernels, native

        native.extension().clear_cache()
        kernels._compile_kernel.cache_clear()
        cases = 0
        largest_error = 0.0
        with patch.object(ada, "schedule", wraps=ada.schedule) as selected:
            for dim in (16, 64, 128, 256, 512):
                for causal in (False, True):
                    for queries, keys in (
                        (127, 113),
                        (128, 128),
                        (128, 129),
                        (129, 128),
                        (1023, 1023),
                        (1025, 1023),
                        (1023, 1031),
                        (1024, 1024),
                        (1025, 1031),
                    ):
                        with self.subTest(dim=dim, causal=causal, queries=queries, keys=keys):
                            q = torch.randn(
                                1, 16, queries, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
                            )
                            k, v = [
                                torch.randn(
                                    1, 16, keys, dim, device="cuda", dtype=q.dtype, requires_grad=True
                                )
                                for _ in range(2)
                            ]
                            dout = torch.randn_like(q)
                            out = api.scaled_dot_product_attention(q, k, v, is_causal=causal)
                            actual = (out, *torch.autograd.grad(out, (q, k, v), dout))
                            errors = self.assert_centered_reference(actual, (q, k, v), dout, causal)
                            largest_error = max(largest_error, *errors)
                            cases += 1
            forward_dimensions = {
                call.args[2] for call in selected.call_args_list if call.args[1] == "forward"
            }
            self.assertEqual(forward_dimensions, {16, 64, 128, 256, 512})
        print(json.dumps({"ada_boundary_cases": cases, "largest_relative_error": largest_error}), flush=True)

    def test_rtx4090_packed_statistics_graph_replay(self):
        if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 4090":
            self.skipTest("Requires the full RTX 4090 dispatch predicate.")
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = [
                    torch.randn(1, 16, 1024, 128, device="cuda", dtype=dtype, requires_grad=True)
                    for _ in range(3)
                ]
                dout = torch.randn_like(inputs[0])

                def run():
                    out = api.scaled_dot_product_attention(*inputs, is_causal=True, scale=-0.25)
                    return out, *torch.autograd.grad(out, inputs, dout)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run()
                torch.cuda.current_stream().wait_stream(stream)
                for replay in range(2):
                    if replay:
                        with torch.no_grad():
                            for tensor in inputs:
                                tensor.copy_(torch.randn_like(tensor) * 0.75)
                    graph.replay()
                    self.assert_centered_reference(actual, inputs, dout, True, scale=-0.25)
                out = api.scaled_dot_product_attention(*inputs, is_causal=True, scale=0.0)
                actual = (out, *torch.autograd.grad(out, inputs, dout))
                self.assert_centered_reference(actual, inputs, dout, True, scale=0.0)
                with torch.no_grad():
                    for tensor in (*inputs, dout):
                        tensor.zero_()
                    inputs[0][..., 0] = 10
                    inputs[1][..., 0] = 9 if dtype == torch.float16 else 5
                    inputs[1][:, :, 0, 0] = 10
                    inputs[2][..., 0], inputs[2][:, :, 0, 0] = -1, 1
                    dout[..., 0] = 1
                out = api.scaled_dot_product_attention(*inputs, is_causal=True, scale=1.0)
                actual = (out, *torch.autograd.grad(out, inputs, dout))
                self.assert_centered_reference(actual, inputs, dout, True, scale=1.0)
                self.assertGreater(actual[1].double().norm().item(), 0.0)
                self.assertGreater(actual[2].double().norm().item(), 0.0)

    def test_real_comfyui_model_patcher_and_attention_hook(self):
        if COMFYUI_ROOT is None:
            self.skipTest("Pass --comfyui-root to check the real ComfyUI integration.")
        sys.path.insert(0, str(COMFYUI_ROOT))
        from comfy.ldm.modules.attention import AttentionTensorContainer, attention_pytorch
        from comfy.model_patcher import ModelPatcher
        from comfy.patcher_extension import WrapperExecutor, WrappersMP

        from cute_attention.nodes import CuTeAttention

        source = ModelPatcher(torch.nn.Linear(8, 8), torch.device("cuda"), torch.device("cpu"))
        patched = CuTeAttention().patch(source, scope="model", mode="strict")[0]
        self.assertNotIn("optimized_attention_override", source.model_options["transformer_options"])
        q = torch.randn(1, 65, 256, device="cuda", dtype=torch.float16)
        expected = attention_pytorch(q, q, q, 4)
        options = patched.model_options["transformer_options"]
        actual = attention_pytorch(q, q, q, 4, transformer_options=options)
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.005)
        containers = tuple(AttentionTensorContainer(q) for _ in range(3))
        actual = attention_pytorch(*containers, 4, transformer_options=options)
        torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.005)
        calls = patched.get_all_wrappers(WrappersMP.APPLY_MODEL)
        q_heads = q.reshape(1, 65, 4, 64).transpose(1, 2)
        executor = WrapperExecutor.new_executor(
            lambda: (runtime._model_mode.get(), F.scaled_dot_product_attention(q_heads, q_heads, q_heads)),
            calls,
        )
        active_mode, direct = executor.execute()
        self.assertEqual(active_mode, "strict")
        torch.testing.assert_close(
            direct.transpose(1, 2).reshape(1, 65, 256), expected, rtol=0.005, atol=0.005
        )
        self.assertIsNone(runtime._model_mode.get())


def main():
    global COMFYUI_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path)
    args = parser.parse_args()
    COMFYUI_ROOT = args.comfyui_root
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GPUValidation)
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
