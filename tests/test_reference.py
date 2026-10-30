import unittest

import torch

from tests.reference import fp64_reference


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
