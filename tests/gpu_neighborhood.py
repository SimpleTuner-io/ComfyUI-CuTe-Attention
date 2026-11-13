"""CUDA neighborhood adapter qualification against independent FP64 and NATTEN."""

import json
import unittest

import torch

from cute_attention.neighborhood import NeighborhoodBackend, na2d


def reference(q, k, v, window, scale):
    batch, height, width, heads, dim = q.shape
    wh, ww = (window, window) if isinstance(window, int) else window
    # Independently enumerate every query's shifted window, including edges.
    mask = torch.zeros(height * width, height * width, dtype=torch.bool, device=q.device)
    for y in range(height):
        for x in range(width):
            y0 = min(max(y - wh // 2, 0), height - wh)
            x0 = min(max(x - ww // 2, 0), width - ww)
            for ky in range(y0, y0 + wh):
                for kx in range(x0, x0 + ww):
                    mask[y * width + x, ky * width + kx] = True
    qf, kf, vf = [t.permute(0, 3, 1, 2, 4).reshape(batch, heads, height * width, dim) for t in (q, k, v)]
    scores = (qf @ kf.transpose(-1, -2)) * scale
    out = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vf
    return out.reshape(batch, heads, height, width, dim).permute(0, 2, 3, 1, 4)


class NeighborhoodGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("This suite requires a CUDA GPU; CPU execution is not GPU validation.")
        torch.manual_seed(123)

    def test_forward_and_backward_against_fp64(self):
        for dtype in (torch.float16, torch.bfloat16):
            for window, scale in ((3, 0.125), ((4, 2), 1.0), ((3, 5), -0.25), (3, 0.0)):
                with self.subTest(dtype=dtype, window=window, scale=scale):
                    operands = [
                        torch.randn(1, 9, 7, 2, 64, device="cuda", dtype=dtype)
                        .transpose(1, 2)
                        .requires_grad_()
                        for _ in range(3)
                    ]
                    doubles = [t.detach().double().requires_grad_() for t in operands]
                    grad = torch.randn_like(operands[0])
                    out = na2d(*operands, window, scale=scale)
                    expected = reference(*doubles, window, scale)
                    actuals = (out,) + torch.autograd.grad(out, operands, grad)
                    expected_values = (expected,) + torch.autograd.grad(expected, doubles, grad.double())
                    errors = []
                    for actual, target in zip(actuals, expected_values):
                        error = (actual.double() - target).norm() / target.norm().clamp_min(1e-12)
                        errors.append(error.item())
                        self.assertTrue(torch.isfinite(actual).all())
                        self.assertLess(error.item(), 0.01 if dtype == torch.bfloat16 else 0.005)
                    print(
                        json.dumps(
                            {"dtype": str(dtype), "window": window, "scale": scale, "relative_l2": errors}
                        )
                    )

    def test_natten_matching_inputs(self):
        from natten.functional import na2d as natten_na2d

        for dtype in (torch.float16, torch.bfloat16):
            operands = [torch.randn(1, 32, 40, 3, 64, device="cuda", dtype=dtype) for _ in range(3)]
            actual = na2d(*operands, 7, scale=1.0)
            expected = natten_na2d(*operands, 7, scale=1.0, backend="cutlass-fna")
            error = (actual.float() - expected.float()).norm() / expected.float().norm()
            self.assertLess(error.item(), 0.01 if dtype == torch.bfloat16 else 0.005)
            print(
                json.dumps(
                    {"comparison": "natten-cutlass-fna", "dtype": str(dtype), "relative_l2": error.item()}
                )
            )

    def test_unsupported_dtype_dimension_and_geometry(self):
        for shape, dtype, window in (
            ((1, 7, 9, 2, 64), torch.float32, 3),
            ((1, 7, 9, 2, 32), torch.float16, 3),
            ((1, 7, 9, 2, 64), torch.float16, 11),
        ):
            q = torch.zeros(shape, device="cuda", dtype=dtype)
            with self.assertRaises((ValueError, RuntimeError)):
                na2d(q, q, q, window)

    def test_explicit_precision_preserves_pmrf_activation_dtype(self):
        operands = [torch.randn(1, 7, 9, 2, 64, device="cuda") for _ in range(3)]
        for precision in ("bfloat16", "float16"):
            selected = NeighborhoodBackend("cute", precision)
            actual = selected.na2d(*operands, 3, scale=1.0)
            expected = na2d(*(t.to(selected.dtype) for t in operands), 3, scale=1.0).float()
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(selected.calls, 1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
