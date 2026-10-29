import unittest

import torch

from cute_attention.attention import ComfyAttention
from cute_attention.runtime import _native_sdpa


class AttentionTests(unittest.TestCase):
    def test_layouts_masks_gqa_and_scale(self):
        torch.manual_seed(123)
        backend = ComfyAttention("auto")
        for preshaped in (False, True):
            for head_output in (False, True):
                for gqa in (False, True):
                    q = torch.randn(2, 4, 5, 8)
                    k = torch.randn(2, 2 if gqa else 4, 7, 8)
                    v = torch.randn_like(k)
                    masks = (
                        None,
                        torch.rand(5, 7) > 0.3,
                        torch.rand(2, 5, 7) > 0.3,
                        torch.rand(2, 4, 5, 7) > 0.3,
                        torch.randn(2, 5, 7),
                    )
                    for mask in masks:
                        with self.subTest(
                            preshaped=preshaped,
                            head_output=head_output,
                            gqa=gqa,
                            mask=None if mask is None else (mask.ndim, mask.dtype),
                        ):
                            normalized_mask = mask
                            if mask is not None and mask.ndim == 3:
                                normalized_mask = mask.unsqueeze(1)
                            expected = _native_sdpa(
                                q, k, v, attn_mask=normalized_mask, scale=-0.25, enable_gqa=gqa
                            )
                            if not head_output:
                                expected = expected.transpose(1, 2).reshape(2, 5, 32)
                            inputs = (q, k, v)
                            if not preshaped:
                                inputs = tuple(t.transpose(1, 2).reshape(2, t.shape[2], -1) for t in inputs)
                            actual = backend(
                                *inputs,
                                4,
                                mask=mask,
                                skip_reshape=preshaped,
                                skip_output_reshape=head_output,
                                scale=-0.25,
                                enable_gqa=gqa,
                            )
                            torch.testing.assert_close(actual, expected)

    def test_noncontiguous_inputs_and_all_hidden_rows(self):
        q = torch.randn(2, 5, 4, 8).transpose(1, 2)
        k = torch.randn(2, 7, 4, 8).transpose(1, 2)
        v = torch.randn_like(k)
        mask = torch.ones(2, 5, 7, dtype=torch.bool)
        mask[:, 1] = False
        out = ComfyAttention("auto")(q, k, v, 4, mask=mask, skip_reshape=True)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(out[:, 1].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
