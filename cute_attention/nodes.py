"""The MODEL patch node and its process-wide SDPA control."""

from comfy.patcher_extension import WrappersMP

from .attention import ComfyAttention
from .runtime import ModelSDPAOverride, install_dispatcher, set_process_mode


class CuTeAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "scope": (
                    ["process", "model"],
                    {
                        "default": "process",
                        "tooltip": "Process affects subsequent Torch SDPA calls throughout ComfyUI. "
                        "Model limits direct Torch calls to this MODEL's apply_model execution.",
                    },
                ),
                "mode": (
                    ["auto", "strict"],
                    {
                        "default": "auto",
                        "tooltip": "Auto uses native SDPA for unsupported inputs. Strict requires CuTe.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = "Use vendored KohakuFA-CuTe attention with optional process-wide Torch SDPA replacement."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Reapply the requested process state when a prompt is queued again.
        return float("nan")

    def patch(self, model, enabled=True, scope="process", mode="auto"):
        if scope not in ("process", "model"):
            raise ValueError(f"Unknown CuTe attention scope: {scope}")
        if mode not in ("auto", "strict"):
            raise ValueError(f"Unknown CuTe attention mode: {mode}")
        if not enabled:
            if scope == "process":
                set_process_mode(None)
            return (model,)

        patched = model.clone()
        install_dispatcher()
        patched.set_model_optimized_attention(ComfyAttention(mode))
        patched.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, "cute_attention")
        patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, "cute_attention", ModelSDPAOverride(mode))
        if scope == "process":
            set_process_mode(mode)
        return (patched,)
