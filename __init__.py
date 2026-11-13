"""ComfyUI entry point. Loading the extension does not patch Torch."""

from .cute_attention.nodes import CuTeAttention
from .cute_attention.pmrf_node import CuTeNeighborhoodAttention

NODE_CLASS_MAPPINGS = {"CuTeAttention": CuTeAttention, "CuTeNeighborhoodAttention": CuTeNeighborhoodAttention}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CuTeAttention": "CuTe Attention (KohakuFA)",
    "CuTeNeighborhoodAttention": "CuTe Neighborhood Attention (PMRF)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
