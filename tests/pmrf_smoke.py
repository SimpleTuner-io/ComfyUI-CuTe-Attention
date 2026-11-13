"""Run the example's real ComfyUI nodes and PMRF weights with both NA backends.

Store outputs outside the repository. The first run of each backend includes
warm-up/compilation; subsequent timings include the complete restoration node.
"""

import argparse
import asyncio
import importlib
import json
import math
import platform
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--input", required=True, help="Filename inside ComfyUI's input directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.comfyui_root))
    import folder_paths
    import nodes

    folder_paths.set_output_directory(str(args.output))
    for root in (args.comfyui_root / "custom_nodes/ComfyUI-PMRF", Path(__file__).resolve().parents[1]):
        if not asyncio.run(nodes.load_custom_node(str(root))):
            raise RuntimeError("ComfyUI custom-node load failed")
    node_type = nodes.NODE_CLASS_MAPPINGS["CuTeNeighborhoodAttention"]
    module = importlib.import_module(node_type.__module__)
    _, model, block_type = module.resolve_pmrf()
    blocks = [b for b in model.modules() if isinstance(b, block_type)]
    original_forwards = [b.forward for b in blocks]
    images = nodes.LoadImage().load_image(args.input)[0]
    observations = []
    backend_type = module.NeighborhoodBackend

    class ObservedBackend(backend_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observations.append(self)

    report = {
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "steps": args.steps,
        "seed": 123,
        "precision": "bfloat16",
        "input_shape": list(images.shape),
        "neighborhood_blocks": len(blocks),
        "timing_boundary": "complete PMRF restoration node; CUDA synchronized wall time",
        "image_agreement_gate": {"max_mean_absolute_error": 2 / 255, "min_psnr_db": 35},
        "runs": [],
    }
    outputs = {}
    with patch.object(module, "NeighborhoodBackend", ObservedBackend):
        for backend in ("natten", "cute"):
            for repetition in range(args.repeats + 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = node_type().restore(images, 1.0, args.steps, 123, "lanczos4", "bfloat16", backend)[0]
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                selected = observations[-1]
                if not selected.calls:
                    raise AssertionError("No face restoration/neighborhood kernel executed")
                if not torch.isfinite(output).all():
                    raise AssertionError("Nonfinite restoration output")
                if [b.forward for b in blocks] != original_forwards:
                    raise AssertionError("PMRF forwards were not restored")
                report["runs"].append(
                    {
                        "backend": backend,
                        "phase": "first_call" if repetition == 0 else "warm",
                        "seconds": elapsed,
                        "calls": selected.calls,
                        "shapes": sorted(selected.shapes),
                        "output_shape": list(output.shape),
                    }
                )
                (args.output / "pmrf-results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report["runs"][-1]), flush=True)
            outputs[backend] = output
            nodes.SaveImage().save_images(output, filename_prefix=f"pmrf-{backend}")
    difference = outputs["cute"] - outputs["natten"]
    mse = difference.square().mean().item()
    report["comparison"] = {
        "mean_absolute_error": difference.abs().mean().item(),
        "max_absolute_error": difference.abs().max().item(),
        "psnr_db": -10 * math.log10(max(mse, 1e-20)),
        "note": "One example and seed; not a perceptual quality qualification or FP32 parity claim.",
    }
    (args.output / "pmrf-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison"]), flush=True)
    if report["comparison"]["mean_absolute_error"] > 2 / 255 or report["comparison"]["psnr_db"] < 35:
        raise AssertionError("CuTe/NATTEN restored images exceeded the declared agreement gate")


if __name__ == "__main__":
    main()
