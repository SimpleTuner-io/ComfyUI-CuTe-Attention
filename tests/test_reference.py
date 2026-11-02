import unittest

import torch

from tests.reference import fp64_reference, rounded_reference


class ReferenceTests(unittest.TestCase):
    def test_centered_equations_match_fp64_autograd(self):
        torch.manual_seed(481)
        for causal in (False, True):
            for scale in (0.125, -0.25, 0.0):
                with self.subTest(causal=causal, scale=scale):
                    q = torch.randn(2, 4, 5, 8, dtype=torch.float64, requires_grad=True)
                    k, v = [torch.randn(2, 4, 7, 8, dtype=q.dtype, requires_grad=True) for _ in range(2)]
                    logits = (q @ k.transpose(-1, -2)) * scale
                    if causal:
                        visible = torch.arange(7)[None, :] <= torch.arange(5)[:, None]
                        logits = logits.masked_fill(~visible, -torch.inf)
                    out = logits.softmax(-1) @ v
                    dout = torch.randn_like(out)
                    expected = (out, *torch.autograd.grad(out, (q, k, v), dout))
                    actual = fp64_reference((q, k, v), dout, causal=causal, scale=scale)
                    for got, wanted in zip(actual, expected):
                        torch.testing.assert_close(got, wanted, rtol=1e-12, atol=1e-12)


class RoundedReferenceTests(unittest.TestCase):
    def test_sharp_attention_preserves_operand_rounding(self):
        q = torch.tensor([[[[10.0]]]], dtype=torch.bfloat16)
        k = torch.tensor([[[[10.0], [5.0]]]], dtype=q.dtype)
        v = torch.tensor([[[[1.0], [-1.0]]]], dtype=q.dtype)
        expected = rounded_reference((q, k, v), torch.ones_like(q), scale=1)
        self.assertGreater(expected[2][0, 0, 1, 0].abs().item(), 0)
        self.assertEqual(expected[2][0, 0, 0, 0].item(), 0)
        self.assertEqual(expected[3][0, 0, 0, 0].item(), 1)

    def test_hidden_rows_and_zero_scale_with_gqa(self):
        torch.manual_seed(17)
        q = torch.randn(1, 4, 5, 8, dtype=torch.bfloat16)
        k, v = [torch.randn(1, 2, 7, 8, dtype=q.dtype) for _ in range(2)]
        mask = torch.ones(5, 7, dtype=torch.bool)
        mask[0] = False
        output = rounded_reference((q, k, v), torch.ones_like(q), mask, scale=0)
        self.assertTrue(all(torch.isfinite(t).all() for t in output))
        self.assertEqual(output[0][:, :, 0].count_nonzero().item(), 0)
        self.assertEqual(output[1].count_nonzero().item(), 0)
        self.assertEqual(output[2].count_nonzero().item(), 0)
        self.assertEqual(output[3].shape, v.shape)

    def test_restores_matmul_precision(self):
        previous = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            q = torch.ones(1, 1, 2, 8, dtype=torch.bfloat16)
            rounded_reference((q, q, q), q)
            self.assertEqual(torch.backends.cuda.matmul.fp32_precision, "tf32")
        finally:
            torch.backends.cuda.matmul.fp32_precision = previous
