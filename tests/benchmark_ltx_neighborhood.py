"""Paired complete na3d calls on LTX shapes; not whole-decoder timings."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F

from cute_attention.vendor.kohakufa_cute import neighborhood_attention_3d


def sampled_reference(q, k, v, window, indices):
    spatial = q.shape[1:4]
    window = tuple(min(w, n) for w, n in zip(window, spatial))
    result = []
    for index in indices:
        z, remainder = divmod(index, spatial[1] * spatial[2])
        y, x = divmod(remainder, spatial[2])
        starts = [max(0, min(c - w // 2, n - w)) for c, w, n in zip((z, y, x), window, spatial)]
        slices = tuple(slice(s, s + w) for s, w in zip(starts, window))
        keys, values = [t[(slice(None), *slices)].flatten(1, 3).double() for t in (k, v)]
        scores = torch.einsum("bhd,bnhd->bhn", q[:, z, y, x].double(), keys)
        result.append(torch.einsum("bhn,bnhd->bhd", scores.softmax(-1), values))
    return torch.stack(result, dim=1)


def measure(function, seconds):
    torch.cuda.synchronize()
    start = time.perf_counter()
    count = 0
    while True:
        function()
        torch.cuda.synchronize()
        count += 1
        elapsed = time.perf_counter() - start
        if elapsed >= seconds:
            return elapsed * 1000 / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--sample-seconds", type=float, default=0.2)
    args = parser.parse_args()
    if args.samples < 1 or args.sample_seconds <= 0:
        parser.error("samples and sample duration must be positive")
    import comfy_kitchen

    torch.set_num_threads(4)
    torch.manual_seed(1729)
    cases = [
        ((7, 8, 12), 32, (3, 7, 7)),
        ((7, 16, 24), 16, (3, 7, 7)),
        ((13, 16, 24), 8, (3, 5, 5)),
        ((25, 32, 48), 8, (3, 5, 5)),
        ((33, 64, 96), 4, (11, 11, 11)),
        ((17, 128, 128), 4, (11, 11, 11)),
        ((13, 64, 65), 2, (11, 11, 11)),
    ]
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
        "method": "CUDA-synchronized complete eager calls; alternating samples after three warm calls; per-call synchronization included",
        "sample_seconds": args.sample_seconds,
        "samples": args.samples,
        "scale": 1.0,
        "reference": "32 sampled queries plus corners and plane seams, independent native FP64 on identical rounded inputs",
        "rows": [],
    }
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16):
            for spatial, heads, window in cases:
                q = (F.normalize(torch.randn(1, *spatial, heads, 64, device="cuda"), dim=-1)).to(dtype)
                k = (F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1) * 8).to(dtype)
                v = torch.randn_like(q)
                functions = {
                    "cute": lambda: neighborhood_attention_3d(q, k, v, window, None, 1.0),
                    "comfy": lambda: comfy_kitchen.na3d(q, k, v, window, None, 1.0),
                }
                length = spatial[0] * spatial[1] * spatial[2]
                plane = spatial[1] * spatial[2]
                indices = sorted(
                    set([0, length - 1, plane - 1, plane] + random.Random(1729).sample(range(length), 32))
                )
                expected = sampled_reference(q, k, v, window, indices)
                errors = {}
                for name, function in functions.items():
                    for _ in range(3):
                        out = function()
                    assert torch.isfinite(out).all()
                    selected = out.flatten(1, 3)[:, indices].double()
                    error = (
                        (
                            (selected - expected).square().sum((0, 1, 3)).sqrt()
                            / expected.square().sum((0, 1, 3)).sqrt().clamp_min(1e-12)
                        )
                        .max()
                        .item()
                    )
                    if error >= (0.005 if dtype == torch.float16 else 0.01):
                        raise AssertionError("Sampled FP64 gate failed")
                    errors[name] = error
                times = {name: [] for name in functions}
                for sample in range(args.samples):
                    for name in (("cute", "comfy") if sample % 2 == 0 else ("comfy", "cute")):
                        times[name].append(measure(functions[name], args.sample_seconds))
                row = {
                    "shape": list(q.shape),
                    "window": window,
                    "dtype": str(dtype),
                    "milliseconds": times,
                    "median_ms": {name: statistics.median(values) for name, values in times.items()},
                    "worst_head_sampled_fp64_relative_l2": errors,
                }
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
