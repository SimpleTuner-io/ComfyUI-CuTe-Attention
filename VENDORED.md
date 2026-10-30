# Vendored KohakuFA-CuTe

Source: <https://github.com/SimpleTuner-io/KohakuFA-CuTe>

Revision: `d4700f581ed538ae72928872e3a1a168cad01c30`

The 2026-10-30 refresh adds upstream RTX 4090 adaptive scheduling, short D128
forward support, fused backward centering, and packed statistics workspaces.
It retains the previous GB10/SM121 optimizations. Dependency requirements and
licenses are unchanged. Existing RTX 4090 measurements describe the earlier
revision identified in `results/README.md`.

The tracked contents of upstream `src/kohakufa_cute` are included in
`cute_attention/vendor/kohakufa_cute`. No external KohakuFA-CuTe install or source
checkout is required. Torch, CUTLASS DSL, TVM FFI, Triton, Ninja, and a C++17
compiler remain runtime dependencies for CUDA kernels.

Local integration changes:

- Torch custom operators use the `comfyui_cute_attention` namespace.
- The native launcher imports its Python module through a build definition
  containing the actual vendored module name, rather than a top-level package.
- Native build and TVM FFI registration names are isolated from upstream. The
  build hash also includes the Python module name to separate import locations.
- Native fallback captures Torch's built-in SDPA directly, so another extension's
  Python monkeypatch cannot become the fallback or cause recursive dispatch.
- The packed-statistics eligibility probe uses Torch's device-properties API.
  This avoids binding to CUDA 12's `cudaGetDeviceProperties_v2` when the compiler
  headers supplied by Triton are used with a CUDA 13 Torch runtime. Eligibility
  remains cached per device, and the GIL is acquired without holding the cache
  mutex to preserve lock ordering.

Kernel arithmetic and automatic backend selection are unchanged. Keep future
updates traceable to an upstream revision and preserve these integration changes.

The root `LICENSE` is the repository's upstream AGPL-3.0 license. `NOTICE` and
`LICENSES/BSD-3-Clause.txt` are retained from KohakuFA-CuTe. The implementation is
AGPL-3.0-or-later; adapted NVIDIA CUTLASS and FlashAttention helpers retain their
BSD notices. The privacy checker, hooks, and their tests are also copied from the
same KohakuFA-CuTe revision.

The centered FP64 reference in `tests/reference.py` is adapted from upstream
`benchmarks/reference.py` at this revision for RTX 4090 schedule qualification.
