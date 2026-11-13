"""Optional PMRF restoration node; importing this module loads no PMRF models."""

import importlib
import logging

from .neighborhood import NeighborhoodBackend, pmrf_neighborhood

log = logging.getLogger(__name__)


def resolve_pmrf():
    import nodes

    node_type = nodes.NODE_CLASS_MAPPINGS.get("PMRF")
    if node_type is None:
        raise RuntimeError("Install and load 2kpr/ComfyUI-PMRF and its model weights before using this node.")
    module = importlib.import_module(node_type.__module__)
    architecture = importlib.import_module(f"{module.__package__}.arch.hourglass.image_transformer_v2")
    return node_type, module.pmrf, architecture.NeighborhoodSelfAttentionBlock


class CuTeNeighborhoodAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 40.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 25, "min": 1, "max": 400}),
                "seed": ("INT", {"default": 123, "min": 0, "max": 2**32}),
                "interpolation": (
                    ["lanczos4", "nearest", "linear", "cubic", "area", "linear_exact", "nearest_exact"],
                ),
                "precision": (
                    ["bfloat16", "float16"],
                    {"tooltip": "Round neighborhood Q/K/V only; preserve the surrounding PMRF computation."},
                ),
                "backend": (
                    ["cute", "natten"],
                    {"tooltip": "CuTe is strict with no silent fallback. NATTEN is an optional comparison."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "restore"
    CATEGORY = "image/restoration"
    DESCRIPTION = (
        "PMRF restoration using KohakuFA-CuTe 2D neighborhood attention. Requires ComfyUI-PMRF, "
        "its weights, and a supported CUDA GPU. Only this execution uses the selected backend."
    )

    def restore(self, images, scale, num_steps, seed, interpolation, precision="bfloat16", backend="cute"):
        selected = NeighborhoodBackend(backend, precision)
        node_type, model, block_type = resolve_pmrf()
        with pmrf_neighborhood(model, block_type, selected):
            node = node_type()
            result = getattr(node, node.FUNCTION)(images, scale, num_steps, seed, interpolation)
        # No detected face is valid PMRF behavior, but is not a kernel validation.
        if not selected.calls:
            log.warning("PMRF made no neighborhood-attention calls; the input may contain no detected face.")
        else:
            log.info("PMRF neighborhood backend=%s precision=%s calls=%d", backend, precision, selected.calls)
        return result
