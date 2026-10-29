"""Warm attention-call timings: python -m tools.benchmark_attention."""

import argparse
import json
from statistics import median

import torch
import torch.nn.functional as F

from cute_attention import runtime
from cute_attention.vendor.kohakufa_cute import api


def timing(function, tensors, iterations, repeats):
    for _ in range(5):
        function(*tensors)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function(*tensors)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {"median_ms": median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--dim", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not api._runtime_available():
        raise RuntimeError("CUDA and the CuTe runtime dependencies are required.")
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": "bfloat16",
                "batch": 1,
                "heads": 16,
                "causal": False,
                "layout": "BHSD view of BSHD storage",
                "method": "CUDA events; median of repeated means; compilation and warm-up excluded",
                "iterations": args.iterations,
                "repeats": args.repeats,
            }
        ),
        flush=True,
    )
    runtime.set_process_mode("strict")
    try:
        with torch.inference_mode():
            for seq in args.seq:
                for dim in args.dim:
                    tensors = tuple(
                        torch.randn(1, seq, 16, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
                        for _ in range(3)
                    )
                    expected = runtime._native_sdpa(*tensors)
                    actual = F.scaled_dot_product_attention(*tensors)
                    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
                    native = timing(runtime._native_sdpa, tensors, args.iterations, args.repeats)
                    cute = timing(F.scaled_dot_product_attention, tensors, args.iterations, args.repeats)
                    print(
                        json.dumps(
                            {
                                "seq": seq,
                                "dim": dim,
                                "native": native,
                                "cute_process_override": cute,
                                "native_over_cute": native["median_ms"] / cute["median_ms"],
                            }
                        ),
                        flush=True,
                    )
    finally:
        runtime.set_process_mode(None)


if __name__ == "__main__":
    main()
