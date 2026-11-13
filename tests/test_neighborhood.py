import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cute_attention import neighborhood as na
from cute_attention.pmrf_node import CuTeNeighborhoodAttention

natten = SimpleNamespace(value="original")


class Block(torch.nn.Module):
    d_head = 64

    def forward(self, x):
        return natten.value, x


class NeighborhoodTests(unittest.TestCase):
    def test_spatial_layout_round_trip_for_strided_inputs(self):
        q = torch.arange(2 * 9 * 7 * 3 * 64).reshape(2, 9, 7, 3, 64).transpose(1, 2)

        def kernel(qf, kf, vf, **kwargs):
            self.assertEqual(qf.shape, (2, 3, 63, 64))
            self.assertEqual(kwargs, {"spatial_shape": (7, 9), "window_size": (3, 5), "scale": 1.0})
            torch.testing.assert_close(qf[1, 2, 2 * 9 + 4], q[1, 2, 4, 2])
            torch.testing.assert_close(kf, qf + 1)
            return vf

        with patch.object(na, "neighborhood_attention", kernel):
            actual = na.na2d(q, q + 1, q + 2, (3, 5), scale=1.0)
        torch.testing.assert_close(actual, q + 2)

    def test_bad_layout_and_cpu_fail_without_fallback(self):
        q = torch.zeros(1, 3, 4, 2, 64)
        with self.assertRaises(ValueError):
            na.na2d(q, q[:, :2], q, 3)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            na.NeighborhoodBackend("cute", "bfloat16").na2d(q, q, q, 3)
        with self.assertRaises(ValueError):
            na.NeighborhoodBackend("auto", "bfloat16")
        with self.assertRaises(ValueError):
            na.NeighborhoodBackend("cute", "float32")

    def test_scoped_forward_restores_and_other_thread_uses_original(self):
        block = Block()
        backend = SimpleNamespace(value="cute")
        with na.pmrf_neighborhood(block, Block, backend):
            self.assertEqual(block(1), ("cute", 1))
            result = []
            worker = threading.Thread(target=lambda: result.append(block(2)))
            worker.start()
            worker.join()
            self.assertEqual(result, [("original", 2)])
            self.assertEqual(natten.value, "original")
        self.assertNotIn("forward", block.__dict__)
        self.assertEqual(block(3), ("original", 3))

    def test_exception_and_partial_install_restore_forward(self):
        first, second = Block(), Block()
        second.d_head = 32
        model = torch.nn.ModuleList([first, second])
        with self.assertRaises(ValueError):
            with na.pmrf_neighborhood(model, Block, SimpleNamespace(value="cute")):
                self.fail("Unsupported head dimension was accepted")
        self.assertNotIn("forward", first.__dict__)
        with self.assertRaisesRegex(RuntimeError, "inference"):
            with na.pmrf_neighborhood(first, Block, SimpleNamespace(value="cute")):
                raise RuntimeError("inference")
        self.assertEqual(first(0), ("original", 0))

    def test_rejects_existing_forward_override(self):
        block = Block()
        block.forward = lambda x: x
        with self.assertRaisesRegex(RuntimeError, "modified"):
            with na.pmrf_neighborhood(block, Block, SimpleNamespace(value="cute")):
                pass
        self.assertEqual(block(4), 4)

    def test_node_uses_existing_pmrf_contract_and_restores_after_failure(self):
        block = Block()

        class PMRF:
            FUNCTION = "pmrf"

            def pmrf(self, *args):
                self_args.extend(args)
                self_values.append(block(5)[0])
                raise RuntimeError("restoration failed")

        self_args, self_values = [], []
        selected = SimpleNamespace(value="cute", calls=1)
        with patch("cute_attention.pmrf_node.resolve_pmrf", return_value=(PMRF, block, Block)):
            with patch("cute_attention.pmrf_node.NeighborhoodBackend", return_value=selected):
                with self.assertRaisesRegex(RuntimeError, "restoration failed"):
                    CuTeNeighborhoodAttention().restore("image", 1.0, 25, 123, "lanczos4")
        self.assertEqual(self_args, ["image", 1.0, 25, 123, "lanczos4"])
        self.assertEqual(self_values, ["cute"])
        self.assertEqual(block(5), ("original", 5))


if __name__ == "__main__":
    unittest.main()
