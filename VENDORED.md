# Vendored KohakuFA-CuTe

Source: <https://github.com/SimpleTuner-io/KohakuFA-CuTe>

Revision: `49e4ecd75e8af18813657afba6b9f5d82848335d`

The 2026-10-31 refresh adds upstream RTX 3090 Ti, L40S, and RTX 5090 scheduling
and compensated key/value backward weights, including medium/wide gradient
owners and forward-normalizer compensation. It retains the previous RTX 4090
and GB10/SM121 paths. Dependency requirements and licenses are unchanged.
Existing RTX 4090 measurements describe the earlier revision identified in
`results/README.md`.

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

Apart from the integration changes above, kernel arithmetic and automatic
backend selection match the recorded upstream revision. Keep future updates
traceable to an upstream revision and preserve these integration changes.

The root `LICENSE` is the repository's upstream AGPL-3.0 license. `NOTICE` and
`LICENSES/BSD-3-Clause.txt` are retained from KohakuFA-CuTe. The implementation is
AGPL-3.0-or-later; adapted NVIDIA CUTLASS and FlashAttention helpers retain their
BSD notices. The privacy checker, hooks, and their tests are also copied from the
same KohakuFA-CuTe revision.

The centered FP64 reference in `tests/reference.py` is adapted from upstream
`benchmarks/reference.py` at this revision for RTX 4090 schedule qualification.
The precision-policy tests in `tests/test_precision_policy.py` and
`tests/test_weight_precision.py` are retained from this revision with vendored
module names and source paths.
