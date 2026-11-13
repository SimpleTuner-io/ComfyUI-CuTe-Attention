# Vendored KohakuFA-CuTe

Source: <https://github.com/SimpleTuner-io/KohakuFA-CuTe>

Revision: `c6f72ca8d9c9e9cb053cce3722e5f6f2f7aa0704` (0.4.0)

The 2026-11-13 refresh imports all 60 tracked package files, including explicit
2D `neighborhood_attention`, geometry, scheduling, custom operators, and the
normalized dQ/reconstructed dK correction defaults. The new PMRF node adapts
NATTEN's `[B, height, width, heads, D]` layout to the public neighborhood API.
No neighborhood kernel arithmetic is changed locally. The neighborhood custom
operators also use the isolated `comfyui_cute_attention` namespace.
See [PMRF and regression validation](results/neighborhood-pmrf-2026-11-13.md)
for this snapshot's RTX 4090 checks.

The 2026-11-02 refresh adds SM80 support, optional pinned FlashAttention-derived
SM80/Hopper packages, short and shared scheduling, live deterministic-backward
guards, and further Ada, Ampere, Hopper, GB10, and Blackwell tuning. Default
backward arithmetic now rounds P and dS once to input precision, with FP32
accumulation and actual rounded O in the output/dO normalization. Compensation
experiments remain in the source but are not enabled by this node.

All tracked files from upstream `src/kohakufa_cute` are included recursively in
`cute_attention/vendor/kohakufa_cute`. No external KohakuFA-CuTe package or source
checkout is needed. Base CUDA dependencies retain their existing bounds. The
`sm80-fa4` and `hopper-fa4` optional dependency groups reproduce upstream's exact
pins, including FlashAttention revision
`4616f5116d33e3ba3cbb71d582caffcfcb49649c`. Optional entry points also check Python
3.13, Torch 2.15.0/CUDA 13.2, package versions, and installed source hashes.

Local integration changes:

- Torch custom operators and deterministic-backward diagnostics use the
  `comfyui_cute_attention` namespace.
- Vendored Python imports are relative, including the optional nested packages
  and Blackwell register helper.
- The native launcher imports its Python module through a build definition
  containing the actual vendored module name.
- Native build and TVM FFI registration names are isolated from upstream. The
  native source hash also includes the Python module name.
- Native fallback captures Torch's built-in SDPA directly, preventing recursive
  dispatch through another extension's Python patch.
- All nine native device-property probes use Torch's runtime-compatible API.
  This preserves CUDA 13 compatibility with Triton's CUDA 12 headers, which can
  otherwise bind the unavailable `cudaGetDeviceProperties_v2` symbol. Per-device
  eligibility remains cached; cache mutexes are released before acquiring the
  GIL. Warm eligibility checks do not enter Python.

Kernel arithmetic and automatic selection otherwise match the recorded upstream
revision. Preserve these adaptations when updating. The optional training paths
are not blanket ComfyUI inference optimizations; see [usage](README.md).

The root `LICENSE` is this repository's AGPL-3.0 license. Updated `NOTICE` and
all `LICENSES/` files retain upstream BSD notices and FlashAttention author
attribution. Optional external packages keep their own licenses.

`tests/reference.py` retains the earlier centered FP64 mathematical reference
and adds `rounded_reference` from this upstream revision's
`benchmarks/reference.py`. These are separate accuracy and arithmetic oracles.
Determinism and automatic-policy CPU tests are adapted from this revision with
vendored imports. Precision-owner tests remain from the preceding snapshot and
cover retained experimental helpers, not the default precision policy.

Historical benchmark and validation records retain their original revisions.
The preceding integration coverage is recorded in
[the refresh report](results/vendor-sync-2026-11-02.md).
