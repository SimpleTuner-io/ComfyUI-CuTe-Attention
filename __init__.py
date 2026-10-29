"""ComfyUI entry point. Loading the extension does not patch Torch."""

from .cute_attention.nodes import CuTeAttention

NODE_CLASS_MAPPINGS = {"CuTeAttention": CuTeAttention}
NODE_DISPLAY_NAME_MAPPINGS = {"CuTeAttention": "CuTe Attention (KohakuFA)"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
