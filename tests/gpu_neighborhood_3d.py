"""3D attention GPU checks against independently enumerated FP64 windows."""

import itertools
import json
import unittest

import torch

from cute_attention.vendor.kohakufa_cute import neighborhood_attention_3d
from cute_attention.ltx_node import LTXNeighborhoodBackend


def reference(q, k, v, kernel, scale):
    spatial = q.shape[1:4]
    window = tuple(min(s, w) for s, w in zip(spatial, kernel))
    coordinates = list(itertools.product(*(range(s) for s in spatial)))
    mask = torch.zeros(len(coordinates), len(coordinates), dtype=torch.bool, device=q.device)
    for i, query in enumerate(coordinates):
        starts = [max(0, min(c - w // 2, s - w)) for c, w, s in zip(query, window, spatial)]
        for j, key in enumerate(coordinates):
            mask[i, j] = all(a <= c < a + w for c, a, w in zip(key, starts, window))
    qf, kf, vf = [t.flatten(1, 3).transpose(1, 2) for t in (q, k, v)]
    out = (qf @ kf.transpose(-1, -2) * scale).masked_fill(~mask, -torch.inf).softmax(-1) @ vf
    return out.transpose(1, 2).reshape(q.shape)


class Neighborhood3DGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA hardware is required for this suite.")
        torch.manual_seed(1729)

    def test_kitchen_comparison_and_fp32_fallback(self):
        import comfy_kitchen

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            operands = [torch.randn(1, 3, 5, 7, 2, 64, device="cuda", dtype=dtype) for _ in range(3)]
            backend = LTXNeighborhoodBackend(comfy_kitchen, "auto")
            actual = backend.na3d(*operands, (3, 7, 7), None, 1.0)
            expected = comfy_kitchen.na3d(*operands, (3, 7, 7), None, 1.0)
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(actual.device, operands[0].device)
            if dtype == torch.float32:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual((backend.calls, backend.fallback_calls), (0, 1))
                with self.assertRaises(ValueError):
                    LTXNeighborhoodBackend(comfy_kitchen, "cute").na3d(*operands, (3, 7, 7), None, 1.0)
            else:
                error = (actual.float() - expected.float()).norm() / expected.float().norm()
                self.assertLess(error.item(), 0.01 if dtype == torch.bfloat16 else 0.005)
                self.assertEqual((backend.calls, backend.fallback_calls), (1, 0))

    def test_output_and_gradients(self):
        cases = [
            ((3, 5, 7), (3, 5, 5), 1.0),
            ((1, 4, 6), (11, 11, 11), 1.0),
            ((4, 3, 5), (2, 2, 4), -0.25),
            ((2, 3, 4), (3, 7, 7), 0.0),
        ]
        for dtype, (spatial, kernel, scale) in itertools.product((torch.float16, torch.bfloat16), cases):
            with self.subTest(dtype=dtype, spatial=spatial, scale=scale):
                operands = [
                    torch.randn(1, *spatial, 2, 64, device="cuda", dtype=dtype)
                    .transpose(2, 3)
                    .requires_grad_()
                    for _ in range(3)
                ]
                doubles = [t.detach().double().requires_grad_() for t in operands]
                grad = torch.randn_like(operands[0])
                out = neighborhood_attention_3d(*operands, kernel, None, scale)
                expected = reference(*doubles, kernel, scale)
                actuals = (out,) + torch.autograd.grad(out, operands, grad)
                targets = (expected,) + torch.autograd.grad(expected, doubles, grad.double())
                errors = []
                for actual, target in zip(actuals, targets):
                    self.assertTrue(torch.isfinite(actual).all())
                    # Per-head gates prevent one head masking another's error.
                    dims = (0, 1, 2, 3, 5)
                    error = (
                        (
                            (actual.double() - target).square().sum(dims).sqrt()
                            / target.square().sum(dims).sqrt().clamp_min(1e-12)
                        )
                        .max()
                        .item()
                    )
                    self.assertLess(error, 0.01 if dtype == torch.bfloat16 else 0.005)
                    errors.append(error)
                print(
                    json.dumps(
                        {
                            "dtype": str(dtype),
                            "shape": list(operands[0].shape),
                            "kernel": kernel,
                            "scale": scale,
                            "worst_head_relative_l2_O_dQ_dK_dV": errors,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
