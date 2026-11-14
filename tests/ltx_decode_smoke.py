"""Compare complete LTX diffusion VAE decodes of one saved generated latent.

Run after the supplied API workflow. Keep captures outside the repository.
First calls include compilation/loading; warm runs alternate backend order.
"""

import argparse
import asyncio
import importlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.comfyui_root))
    import folder_paths
    import nodes
    import safetensors.torch
    from comfy.ldm.lightricks.vae.na_diffusion_decoder import NeighborhoodAttention3D

    folder_paths.set_output_directory(str(args.output))
    if not asyncio.run(nodes.load_custom_node(str(Path(__file__).resolve().parents[1]))):
        raise RuntimeError("ComfyUI extension load failed")
    node_type = nodes.NODE_CLASS_MAPPINGS["CuTeLTXVAE"]
    module = importlib.import_module(node_type.__module__)
    vae = nodes.VAELoader().load_vae("ltx-2.5-video-vae-bf16.safetensors")[0]
    saved = safetensors.torch.load_file(args.latent, device="cpu")
    if "latent_format_version_0" not in saved:
        raise ValueError("Expected a current ComfyUI SaveLatent output")
    samples = {"samples": saved["latent_tensor"].float()}
    blocks = [b for b in vae.first_stage_model.modules() if isinstance(b, NeighborhoodAttention3D)]
    original = [b.forward for b in blocks]
    observations = []
    backend_type = module.LTXNeighborhoodBackend

    class ObservedBackend(backend_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observations.append(self)

    report = {
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "latent_shape": list(samples["samples"].shape),
        "neighborhood_blocks": len(blocks),
        "timing_boundary": "complete decode node; synchronized wall time, resident weights after first call",
        "gate": {"max_mae": 2 / 255, "min_psnr_db": 35},
        "runs": [],
        "comparisons": [],
    }
    with patch.object(module, "LTXNeighborhoodBackend", ObservedBackend), torch.inference_mode():
        for tiled in (False, True):
            outputs = {}
            for repetition in range(args.repeats + 1 if not tiled else 1):
                order = ("comfy", "cute") if repetition % 2 == 0 else ("cute", "comfy")
                for backend in order:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    out = node_type().decode(samples, vae, backend, tiled, 256, 64, 24, 8)[0]
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    selected = observations[-1]
                    assert selected.calls if backend == "cute" else selected.fallback_calls
                    assert [b.forward for b in blocks] == original
                    assert torch.isfinite(out).all()
                    row = {
                        "backend": backend,
                        "tiled": tiled,
                        "phase": "first_call" if repetition == 0 else "warm",
                        "seconds": elapsed,
                        "cute_calls": selected.calls,
                        "comfy_calls": selected.fallback_calls,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "output_shape": list(out.shape),
                        "shapes": sorted(selected.shapes),
                    }
                    report["runs"].append(row)
                    outputs[backend] = out.cpu()
                    (args.output / "ltx-decode.json").write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
            delta = outputs["cute"].float() - outputs["comfy"].float()
            mae = delta.abs().mean().item()
            psnr = -10 * math.log10(max(delta.square().mean().item(), 1e-20))
            report["comparisons"].append(
                {"tiled": tiled, "mae": mae, "psnr_db": psnr, "max_absolute_error": delta.abs().max().item()}
            )
            (args.output / "ltx-decode.json").write_text(json.dumps(report, indent=2) + "\n")
            for backend, out in outputs.items():
                nodes.SaveImage().save_images(out[[0, len(out) // 2, -1]], f"ltx-{backend}-tiled-{tiled}")
            if mae > 2 / 255 or psnr < 35:
                raise AssertionError("Complete decode exceeded the declared image agreement gate")
    print("LTX complete decode comparison passed", flush=True)


if __name__ == "__main__":
    main()
