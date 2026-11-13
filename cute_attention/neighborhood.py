"""Explicit NATTEN-layout adapter and execution-scoped PMRF integration."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import RLock
from types import FunctionType, MethodType, SimpleNamespace

import torch

from .vendor.kohakufa_cute import neighborhood_attention

_active = ContextVar("cute_neighborhood_session", default=None)
_patch_lock = RLock()
_MISSING = object()


def na2d(q, k, v, kernel_size, *, scale=None):
    """Strict CuTe adapter: [B, H, W, heads, 64], no implicit dtype changes."""
    if q.ndim != 5 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Expected matching [batch, height, width, heads, 64] Q/K/V.")
    batch, height, width, heads, dim = q.shape
    flat = [t.permute(0, 3, 1, 2, 4).reshape(batch, heads, height * width, dim) for t in (q, k, v)]
    out = neighborhood_attention(*flat, spatial_shape=(height, width), window_size=kernel_size, scale=scale)
    return out.reshape(batch, heads, height, width, dim).permute(0, 2, 3, 1, 4)


class NeighborhoodBackend:
    """PMRF's small fused-NA interface; precision is an explicit node setting."""

    def __init__(self, backend, precision):
        if backend not in ("cute", "natten"):
            raise ValueError(f"Unknown neighborhood backend: {backend}")
        if precision not in ("bfloat16", "float16"):
            raise ValueError(f"Unknown neighborhood precision: {precision}")
        self.dtype = getattr(torch, precision)
        self.backend = backend
        self.calls = 0
        self.shapes = set()
        self.function = na2d
        if backend == "natten":
            try:
                from natten.functional import na2d as natten_na2d
            except ImportError as error:
                raise RuntimeError(
                    "The NATTEN comparison backend requires a compatible NATTEN installation."
                ) from error
            self.function = natten_na2d
        self.functional = SimpleNamespace(na2d=self.na2d)

    @staticmethod
    def has_fused_na():
        return True

    def na2d(self, q, k, v, kernel_size, *, scale=None):
        if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)):
            raise ValueError("Neighborhood Q/K/V must share device and dtype.")
        if q.device.type != "cuda":
            raise ValueError("The PMRF neighborhood node requires CUDA.")
        dtype = q.dtype
        # Keep PMRF's projection, cosine scaling, rotary embedding and output
        # projection unchanged. Round only the attention operands, explicitly.
        operands = [t.to(self.dtype) for t in (q, k, v)]
        out = self.function(*operands, kernel_size, scale=scale)
        self.calls += 1
        self.shapes.add((tuple(q.shape), str(dtype), str(self.dtype), str(kernel_size), scale))
        return out.to(dtype)


@contextmanager
def pmrf_neighborhood(model, block_type, backend):
    """Use the original HDiT forward with a private NATTEN binding.

    Instance wrappers are temporary. A context token sends unrelated threads to
    the original forward, and restoration also runs after an inference failure.
    Neither the NATTEN module nor Torch SDPA is patched.
    """
    with _patch_lock:
        blocks = [block for block in model.modules() if isinstance(block, block_type)]
        if not blocks:
            raise ValueError("The PMRF model contains no supported neighborhood-attention blocks.")
        saved = []
        session = object()
        token = _active.set(session)
        try:
            for block in blocks:
                original = block.forward
                function = getattr(original, "__func__", None)
                if function is not block_type.forward or "natten" not in function.__globals__:
                    raise RuntimeError("PMRF neighborhood forward is already modified or unsupported.")
                if block.d_head != 64:
                    raise ValueError("CuTe PMRF neighborhood attention requires head dimension 64.")
                namespace = dict(function.__globals__, natten=backend)
                replacement = FunctionType(
                    function.__code__,
                    namespace,
                    function.__name__,
                    function.__defaults__,
                    function.__closure__,
                )
                replacement.__kwdefaults__ = function.__kwdefaults__
                bound = MethodType(replacement, block)

                def make_dispatch(original, bound):
                    @wraps(original)
                    def dispatch(*args, **kwargs):
                        selected = bound if _active.get() is session else original
                        return selected(*args, **kwargs)

                    return dispatch

                saved.append((block, block.__dict__.get("forward", _MISSING)))
                block.forward = make_dispatch(original, bound)
            yield backend
        finally:
            for block, forward in reversed(saved):
                if forward is _MISSING:
                    del block.forward
                else:
                    block.forward = forward
            _active.reset(token)
