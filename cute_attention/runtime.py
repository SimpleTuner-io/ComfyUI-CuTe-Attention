"""Process and execution-context selection for Torch SDPA."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

import torch.nn.functional as F

from .vendor.kohakufa_cute import (
    automatic_scaled_dot_product_attention,
    scaled_dot_product_attention,
)
from .vendor.kohakufa_cute.api import _native_sdpa

_BACKENDS = {
    "auto": automatic_scaled_dot_product_attention,
    "strict": scaled_dot_product_attention,
}
_model_mode = ContextVar("comfy_cute_attention_mode", default=None)
_process_mode = None
_install_lock = Lock()


def dispatch_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    mode = _model_mode.get() or _process_mode
    sdpa = _native_sdpa if mode is None else _BACKENDS[mode]
    return sdpa(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def install_dispatcher():
    with _install_lock:
        current = F.scaled_dot_product_attention
        if current is dispatch_sdpa:
            return
        if current is not _native_sdpa:
            raise RuntimeError(
                "Another extension has replaced Torch SDPA. Disable that override "
                "and restart ComfyUI before enabling CuTe Attention."
            )
        F.scaled_dot_product_attention = dispatch_sdpa


def set_process_mode(mode):
    global _process_mode
    if mode is not None and mode not in _BACKENDS:
        raise ValueError(f"Unknown CuTe attention mode: {mode}")
    if mode is not None:
        install_dispatcher()
    _process_mode = mode


@contextmanager
def model_mode(mode):
    if mode not in _BACKENDS:
        raise ValueError(f"Unknown CuTe attention mode: {mode}")
    install_dispatcher()
    token = _model_mode.set(mode)
    try:
        yield
    finally:
        _model_mode.reset(token)


class ModelSDPAOverride:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, executor, *args, **kwargs):
        with model_mode(self.mode):
            return executor(*args, **kwargs)
