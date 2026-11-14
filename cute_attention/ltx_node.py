"""Execution-scoped 3D neighborhood attention for ComfyUI's LTX diffusion VAE."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
from threading import RLock
from types import FunctionType, MethodType

import torch

from .vendor.kohakufa_cute import neighborhood_attention_3d
from .vendor.kohakufa_cute.api import _CAPABILITIES, _runtime_available

log = logging.getLogger(__name__)
_active = ContextVar("cute_ltx_decode", default=None)
_lock = RLock()
_MISSING = object()


class LTXNeighborhoodBackend:
    """Proxy the kitchen module, overriding only na3d without changing dtype."""

    def __init__(self, kitchen, backend):
        if backend not in ("cute", "auto", "comfy"):
            raise ValueError("Unknown LTX neighborhood backend.")
        self.kitchen = kitchen
        self.backend = backend
        self.calls = 0
        self.fallback_calls = 0
        self.shapes = set()

    def __getattr__(self, name):
        return getattr(self.kitchen, name)

    @staticmethod
    def supported(q, k, v):
        return (
            all(t.ndim == 6 for t in (q, k, v))
            and q.shape == k.shape == v.shape
            and all(q.shape)
            and q.shape[-1] == 64
            and q.device.type == "cuda"
            and q.device == k.device == v.device
            and q.dtype == k.dtype == v.dtype
            and q.dtype in (torch.float16, torch.bfloat16)
            and torch.cuda.get_device_capability(q.device) in _CAPABILITIES
            and _runtime_available()
        )

    def na3d(self, q, k, v, kernel_size, is_causal=None, scale=None):
        if self.backend == "comfy" or (self.backend == "auto" and not self.supported(q, k, v)):
            self.fallback_calls += 1
            return self.kitchen.na3d(q, k, v, kernel_size, is_causal, scale)
        # Strict API validates geometry and causal flags. Compilation/runtime
        # failures propagate, including in auto mode; they are not fallbacks.
        out = neighborhood_attention_3d(q, k, v, kernel_size, is_causal, scale)
        self.calls += 1
        window = (kernel_size,) * 3 if type(kernel_size) is int else tuple(kernel_size)
        self.shapes.add((tuple(q.shape), str(q.dtype), window, scale))
        return out


@contextmanager
def ltx_neighborhood(model, block_type, backend="cute"):
    """Bind a private kitchen proxy to each original block forward temporarily.

    The original forward retains QKV, RMS/RoPE, pre-scaling and projections.
    Context dispatch isolates unrelated threads; finally restores every block.
    The kitchen module, class methods and Torch SDPA are never replaced.
    """
    with _lock:
        modules = list(model.modules())
        if any(
            getattr(m, "_compiled_call_impl", None) is not None or hasattr(m, "_orig_mod") for m in modules
        ):
            raise RuntimeError("Compiled LTX decoders are unsupported; use the original ComfyUI VAE.")
        blocks = [block for block in modules if isinstance(block, block_type)]
        if not blocks:
            raise ValueError("Use the LTX diffusion video VAE; this VAE has no 3D neighborhood blocks.")
        original_function = block_type.forward
        kitchen = original_function.__globals__.get("comfy_kitchen")
        if kitchen is None or not callable(getattr(kitchen, "na3d", None)):
            raise RuntimeError("Unsupported ComfyUI LTX attention implementation.")
        selected = LTXNeighborhoodBackend(kitchen, backend)
        session = object()
        token = _active.set(session)
        saved = []
        try:
            for block in blocks:
                original = block.forward
                if getattr(original, "__func__", None) is not original_function:
                    raise RuntimeError("LTX neighborhood forward is already modified or unsupported.")
                namespace = dict(original_function.__globals__, comfy_kitchen=selected)
                replacement = FunctionType(
                    original_function.__code__,
                    namespace,
                    original_function.__name__,
                    original_function.__defaults__,
                    original_function.__closure__,
                )
                replacement.__kwdefaults__ = original_function.__kwdefaults__
                bound = MethodType(replacement, block)

                def dispatch_factory(original, bound):
                    @wraps(original)
                    def dispatch(*args, **kwargs):
                        return (bound if _active.get() is session else original)(*args, **kwargs)

                    return dispatch

                saved.append((block, block.__dict__.get("forward", _MISSING)))
                block.forward = dispatch_factory(original, bound)
            yield selected
        finally:
            for block, forward in reversed(saved):
                if forward is _MISSING:
                    del block.forward
                else:
                    block.forward = forward
            _active.reset(token)


class CuTeLTXVAE:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "backend": (["cute", "auto", "comfy"],),
                "tiled": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "model/latent"
    DESCRIPTION = (
        "Decode with KohakuFA-CuTe 3D neighborhood attention in the LTX diffusion video VAE. "
        "cute requires CUDA FP16/BF16; auto preserves ComfyUI for unsupported devices/dtypes; "
        "comfy is the unchanged comparison backend. No implicit precision conversion."
    )

    def decode(
        self,
        samples,
        vae,
        backend="cute",
        tiled=False,
        tile_size=512,
        overlap=64,
        temporal_size=64,
        temporal_overlap=8,
    ):
        import nodes
        from comfy.ldm.lightricks.vae.na_diffusion_decoder import NeighborhoodAttention3D

        with ltx_neighborhood(vae.first_stage_model, NeighborhoodAttention3D, backend) as selected:
            if tiled:
                result = nodes.VAEDecodeTiled().decode(
                    vae, samples, tile_size, overlap, temporal_size, temporal_overlap
                )
            else:
                result = nodes.VAEDecode().decode(vae, samples)
        if not selected.calls and not selected.fallback_calls:
            raise RuntimeError("The decoder executed no supported 3D neighborhood-attention calls.")
        log.info(
            "LTX 3D neighborhood: backend=%s cute_calls=%d comfy_calls=%d",
            backend,
            selected.calls,
            selected.fallback_calls,
        )
        return result
