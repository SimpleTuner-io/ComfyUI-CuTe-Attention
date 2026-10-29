"""Translate ComfyUI attention layouts into the Torch SDPA convention."""

from .runtime import _BACKENDS


class ComfyAttention:
    def __init__(self, mode):
        self.mode = mode

    def __call__(
        self,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        batch = q.shape[0]
        enable_gqa = kwargs.get("enable_gqa", False)
        if not skip_reshape:
            dim = q.shape[-1] // heads
            q = q.reshape(batch, -1, heads, dim).transpose(1, 2)
            key_heads = k.shape[-1] // dim if enable_gqa else heads
            value_heads = v.shape[-1] // dim if enable_gqa else heads
            k = k.reshape(batch, -1, key_heads, dim).transpose(1, 2)
            v = v.reshape(batch, -1, value_heads, dim).transpose(1, 2)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        # Match attention_pytorch: attn_precision does not cast SDPA inputs.
        out = _BACKENDS[self.mode](
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=kwargs.get("scale"),
            enable_gqa=enable_gqa,
        )
        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(batch, -1, heads * out.shape[-1])
