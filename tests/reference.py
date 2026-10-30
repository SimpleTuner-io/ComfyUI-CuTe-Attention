"""Centered FP64 equations from KohakuFA-CuTe's benchmarks/reference.py."""

import torch


@torch.no_grad()
def fp64_reference(inputs, dout, causal=False, scale=None):
    q, k, v = [tensor.double() for tensor in inputs]
    dout = dout.double()
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    logits = (q @ k.transpose(-1, -2)) * scale
    if causal:
        visible = (
            torch.arange(k.shape[2], device=q.device)[None, :]
            <= torch.arange(q.shape[2], device=q.device)[:, None]
        )
        logits.masked_fill_(~visible, -torch.inf)
    probability = torch.softmax(logits, -1)
    dp = dout @ v.transpose(-1, -2)
    centered = dp - dp.gather(-1, logits.argmax(-1, keepdim=True))
    ds = probability * (centered - (probability * centered).sum(-1, keepdim=True))
    return (
        probability @ v,
        (ds @ k) * scale,
        (ds.transpose(-1, -2) @ q) * scale,
        probability.transpose(-1, -2) @ dout,
    )
