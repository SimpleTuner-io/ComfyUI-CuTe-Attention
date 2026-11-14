"""ComfyUI entry point. Loading the extension does not patch Torch."""

from .cute_attention.nodes import CuTeAttention
from .cute_attention.pmrf_node import CuTeNeighborhoodAttention
from .cute_attention.ltx_node import CuTeLTXVAE

NODE_CLASS_MAPPINGS = {
    "CuTeAttention": CuTeAttention,
    "CuTeNeighborhoodAttention": CuTeNeighborhoodAttention,
    "CuTeLTXVAE": CuTeLTXVAE,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CuTeAttention": "CuTe Attention (KohakuFA)",
    "CuTeNeighborhoodAttention": "CuTe Neighborhood Attention (PMRF)",
    "CuTeLTXVAE": "CuTe 3D Neighborhood Decode (LTX VAE)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
