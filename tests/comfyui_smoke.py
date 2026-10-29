"""Load through ComfyUI and launch CuTe using its real custom-node module name."""

import argparse
import asyncio
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui_root))
    import nodes as comfy_nodes
    from comfy.ldm.modules.attention import attention_pytorch
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrapperExecutor, WrappersMP

    original = F.scaled_dot_product_attention
    root = Path(__file__).resolve().parents[1]
    if not asyncio.run(comfy_nodes.load_custom_node(str(root))):
        raise RuntimeError("ComfyUI failed to load the CuTe Attention custom node.")
    assert F.scaled_dot_product_attention is original
    node = comfy_nodes.NODE_CLASS_MAPPINGS["CuTeAttention"]()
    source = ModelPatcher(torch.nn.Linear(8, 8), torch.device("cuda"), torch.device("cpu"))
    q = torch.randn(1, 65, 256, device="cuda", dtype=torch.float16)
    expected = attention_pytorch(q, q, q, 4)
    patched = node.patch(source, enabled=True, scope="process", mode="strict")[0]
    actual = attention_pytorch(q, q, q, 4, transformer_options=patched.model_options["transformer_options"])
    torch.testing.assert_close(actual, expected, rtol=0.005, atol=0.005)
    qh = q.reshape(1, 65, 4, 64).transpose(1, 2)
    direct = F.scaled_dot_product_attention(qh, qh, qh)
    torch.testing.assert_close(direct.transpose(1, 2).reshape_as(q), expected, rtol=0.005, atol=0.005)
    node.patch(source, enabled=False, scope="process")
    torch.testing.assert_close(
        F.scaled_dot_product_attention(qh, qh, qh), original(qh, qh, qh), rtol=0, atol=0
    )
    # The cloned model keeps its own scope after the process default is cleared.
    executor = WrapperExecutor.new_executor(
        lambda: F.scaled_dot_product_attention(qh, qh, qh),
        patched.get_all_wrappers(WrappersMP.APPLY_MODEL),
    )
    torch.testing.assert_close(executor.execute(), direct)
    print("ComfyUI loader, native module import, process override, disable, and model scope: passed.")


if __name__ == "__main__":
    main()
