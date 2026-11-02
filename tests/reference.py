"""Mathematical and rounded-operand oracles adapted from KohakuFA-CuTe."""

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


@torch.no_grad()
def rounded_reference(inputs, dout, mask=None, causal=False, scale=None, *, output=None):
    """Diagnose the single-rounded-P/dS contract, separately from FP64 accuracy.

    Accumulate the final products in FP64 to expose operand rounding without
    adding another reduction-order dependency. Runtime FP32 accumulation still
    has its own rounding floor. Scores and dP use IEEE FP32 GEMMs; their
    reduction order can differ from tensor-core products. Supplying O isolates backward
    from a forward output-rounding difference in delta = sum(O * dO).
    """
    q, k, v = [tensor.double() for tensor in inputs]
    dtype = inputs[0].dtype
    groups = q.shape[1] // k.shape[1]
    keys, values = [tensor.repeat_interleave(groups, dim=1) for tensor in (k, v)]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    precision = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        with torch.autocast(device_type=q.device.type, enabled=False):
            scores = (q.float() @ keys.float().transpose(-1, -2)).double()
            dp = dout.float() @ values.float().transpose(-1, -2)
    finally:
        torch.backends.cuda.matmul.fp32_precision = precision
    # Shift the unscaled scores before multiplication, preserving small gaps.
    signed = scores if scale >= 0 else -scores
    visible = torch.ones_like(scores, dtype=torch.bool)
    if mask is not None:
        visible &= mask
    if causal:
        visible &= (
            torch.arange(k.shape[2], device=q.device)[None]
            <= torch.arange(q.shape[2], device=q.device)[:, None]
        )
    signed = signed.masked_fill(~visible, -torch.inf)
    valid = visible.any(-1, keepdim=True)
    maximum = signed.amax(-1, keepdim=True).masked_fill(~valid, 0)
    weights = torch.exp((signed - maximum) * abs(scale)).masked_fill(~visible, 0)
    probability = weights / weights.sum(-1, keepdim=True).masked_fill(~valid, 1)
    out = probability @ values
    base = (out.to(dtype) if output is None else output).float()
    delta = (base * dout.float()).sum(-1, keepdim=True)
    rounded_p = probability.float().to(dtype).float()
    ds = (rounded_p * (dp - delta)).to(dtype).double()
    dq = (ds @ keys) * scale
    dk = (ds.transpose(-1, -2) @ q) * scale
    dv = rounded_p.double().transpose(-1, -2) @ dout.double()
    dk, dv = [tensor.reshape(k.shape[0], k.shape[1], groups, *k.shape[2:]).sum(2) for tensor in (dk, dv)]
    return out, dq, dk, dv
