# Vendor refresh: 2026-10-30

KohakuFA-CuTe is synced to `d4700f581ed538ae72928872e3a1a168cad01c30`.
This brings in RTX 4090 adaptive scheduling, short D128 forward support,
fused backward centering, and packed backward statistics. The ComfyUI operator
namespace, native module import, and original SDPA fallback are preserved.

The new device probe initially failed to load on CUDA 13: Triton's bundled
headers referenced `cudaGetDeviceProperties_v2`, which the loaded CUDA 13 runtime
did not export. The vendored launcher now queries Torch's device-properties API,
caches eligibility per device, and releases the cache mutex before acquiring
the GIL. Kernel arithmetic and the device eligibility conditions are unchanged.

## Validation

Tested against ComfyUI 0.43.0, revision
`35060f0156181ae78a86b10f098f67ce8e83697f`, on a full RTX 4090 with 128 SMs,
24 GB nominal VRAM, driver 595.91.07, and a 450 W power limit. The GPU environment
used Python 3.12.3, Torch 2.11.0+cu130, CUTLASS DSL 4.8.1, TVM FFI
0.1.15.post2, Triton 3.6.0, and Ninja 1.13.0.

- All 33 CPU unit tests passed on macOS and Linux.
- All seven GPU validation groups passed, covering native fallback, layouts,
  masks, GQA, FP16/BF16 outputs and gradients, compiler capture, and the real
  ComfyUI model-patcher hook.
- A focused boundary run passed 90 BF16 cases across dimensions
  16/64/128/256/512, dense/causal attention, and both query/key sequence axes
  around 128 and 1024 tokens. The test verifies selection of the Ada scheduler
  for every forward head dimension. Centered FP64 comparisons of outputs and
  dQ/dK/dV had a largest L2-relative error of approximately 0.242% across these
  cases.
- The packed D128 causal workspace passed changed-input CUDA graph replay in
  FP16 and BF16, negative and zero scales, and sharp-attention cases requiring
  nonzero query/key gradients.
- The real ComfyUI custom-node loader smoke test passed, including process-wide
  override, disable, and model scope. The built wheel contains the updated
  kernel modules and the adapted C++ source.

These checks qualify attention and its ComfyUI integration hooks. Full generation
with pretrained checkpoints and GB10 execution were not tested in this refresh.
Performance measurements were not refreshed; the earlier RTX 4090 timings remain
tied to the original revision in [the historical report](README.md).

[Machine-readable validation](vendor-sync-2026-10-30.json) records the exact
boundary matrix, runtime versions, and adapted native-source hash. Raw logs are
kept outside the tracked tree.

Reproduce the GPU and loader checks from the repository root:

```sh
python -m tests.gpu_validation --comfyui-root ../ComfyUI
python -m tests.comfyui_smoke --comfyui-root ../ComfyUI
```
