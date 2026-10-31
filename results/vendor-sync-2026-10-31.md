# Vendor refresh: 2026-10-31

KohakuFA-CuTe is synced to `49e4ecd75e8af18813657afba6b9f5d82848335d`.
This adds nine modules for RTX 3090 Ti, L40S, and RTX 5090 scheduling,
compensated key/value gradient weights, and medium/wide backward ownership.
The ComfyUI namespace, native module isolation, native SDPA fallback, and
CUDA 13 device-probe adaptation are retained. Dependencies and licenses are
unchanged.

## Validation

All 44 CPU tests passed on macOS and Linux. These include 11 imported upstream
policy tests covering shared owner replacement, preservation of query and
custom/masked owners, explicit compile-device selection, precision flags, and
full-CTA barrier placement outside partial-thread guards.

All nine new modules imported with CUDA dependencies installed, and the built
wheel contains all 25 vendored files, including the adapted native source.

The existing seven GPU validation groups and the real ComfyUI custom-node loader
smoke test passed on a full RTX 4090 with 128 SMs and a 350 W power limit. The
GPU environment used Python 3.12.3, Torch 2.11.0+cu130, driver 580.142, CUTLASS
DSL 4.8.1, TVM FFI 0.1.15.post2, Triton 3.6.0, and Ninja 1.13.0. ComfyUI was
0.43.0 at `35060f0156181ae78a86b10f098f67ce8e83697f`.

GPU checks included 90 BF16 boundary cases against centered FP64 equations,
FP16/BF16 outputs and gradients, masks, GQA, native fallback, changed-input graph
replay, saturated gradients, compiler capture, process override/disable, and
model-scoped attention.

**The new RTX 3090 Ti, L40S, and RTX 5090 device-specific kernels were not
GPU-tested in this integration.** Their CPU policy checks and successful imports
do not establish GPU numerical qualification. Full generation with pretrained
checkpoints was not tested. Performance measurements were not refreshed; earlier
timings remain tied to their original source revisions.

[Machine-readable validation](vendor-sync-2026-10-31.json) records the runtime,
hardware coverage, boundary error summary, and adapted native-source hash. Raw
logs remain outside the tracked tree.

Reproduce the integration checks from the repository root:

```sh
python -m unittest discover -v
python -m tests.gpu_validation --comfyui-root ../ComfyUI
python -m tests.comfyui_smoke --comfyui-root ../ComfyUI
```
