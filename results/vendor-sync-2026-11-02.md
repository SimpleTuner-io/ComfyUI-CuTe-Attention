# Vendor refresh: 2026-11-02

KohakuFA-CuTe is synced to release 0.4.0 at
`30704427809008e83ad38174e7c1780016c44258`. All 56 tracked source files are
vendored recursively, including the optional FlashAttention-derived packages.
The node retains isolated operator/FFI/build names, built-in Torch SDPA fallback,
relative imports, and the CUDA 13 device-property adaptation. The new optional
extras mirror upstream pins and include the added FlashAttention BSD notices.

The documentation audit updates installation, supported capabilities, optional
runtime qualification, process/model scope behavior, precision, determinism,
and license attribution. The node's inputs and default scope are unchanged.

## Arithmetic and measurement scope

Default backward now rounds P and dS once to input precision, with FP32
accumulation and actual rounded O in delta. Retained compensation experiments
are not selected by the node. Boundary and graph tests use upstream's rounded
operand oracle with the actual forward output; separate FP64 comparisons still
measure mathematical accuracy. Tolerances were not loosened to accommodate the
new policy.

The new automatic RTX 4090 short/shared paths require gradients and the exact
optional stack. Inference controls verify that contiguous inputs without
gradients keep ordinary CuTe dispatch. Training performance reported upstream
must not be treated as an end-to-end ComfyUI inference speedup.

## Validation

All 52 CPU tests passed on macOS with Python 3.13.18/Torch 2.15.0 and on the
Linux GPU host under both Python 3.12 and 3.13. The source distribution and
wheel build successfully; the wheel includes all 56 vendored files and license
notices. All 53 vendored Python submodules import with the optional stack.

The base environment passed seven GPU groups with two optional groups skipped,
plus the actual ComfyUI custom-node loader smoke test. Coverage includes
FP16/BF16 outputs and gradients, masks and all-hidden rows, GQA, strided layouts,
negative/zero scales, native fallback, model/process scope and disable, changed
input CUDA graph replay, and direct vendored full-graph Dynamo capture using
its eager backend. The 90 BF16 Ada boundary cases had a largest relative error
of 0.002236 against the rounded operand reference. The ten mathematical cases
had maximum gradient relative errors of 0.000385 (FP16) and 0.003057 (BF16)
against Torch FP64 SDPA. These bounds describe the recorded inputs.

The qualified environment passed all nine GPU groups and the real loader smoke
test. It adds 20 FP16/BF16 tuned training cases (five shape families, dense and
causal), each checked through both strict and automatic APIs against the rounded
operand reference. Three contiguous inference controls exclude those training
routes. Live backward policy rejects unordered atomics in strict deterministic
mode, warns in warning-only mode, and selects repeatable ordinary CuTe when
strict determinism is enabled before forward. Its 90 boundary cases had maximum
relative error 0.002229; mathematical gradient errors were at most 0.000367
(FP16) and 0.003130 (BF16).

Both environments use a full RTX 4090 (128 SMs, 450 W limit), driver 595.99.02,
Torch 2.15.0+cu132, CUDA 13.2, CUTLASS DSL 4.8.1, TVM FFI 0.1.15.post0,
Triton 3.9.0, and Ninja 1.13.2. The base environment uses Python 3.12.3 without
FlashAttention 4. ComfyUI is 0.43.0 at
`35060f0156181ae78a86b10f098f67ce8e83697f`.

The qualified environment uses Python 3.13.18, quack-kernels 0.6.5,
torch-c-dlpack-ext 0.1.5, and FlashAttention 4
`4.0.0b38.dev3+g4616f51` from the pinned source. Its source fingerprints pass
upstream's availability guard.

## Warm inference timings

These are new measurements of this snapshot on the qualified environment, with
no gradients: B1/H16, BF16 dense attention, independent Q/K/V, BHSD views of BSHD
storage. Five warm-up calls per backend precede CUDA-event timing; compilation
and warm-up are excluded. Each sample averages 30 calls, and the table reports
the median of five samples. Native uses Torch's default SDPA backend selection;
CuTe includes the persistent process dispatcher and ordinary CuTe kernels.

| Sequence | Head dim | Native SDPA (ms) | CuTe override (ms) | Native / CuTe |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 0.0353 | 0.0360 | 0.98 |
| 1024 | 128 | 0.0676 | 0.0640 | 1.06 |
| 1024 | 256 | 0.1282 | 0.1264 | 1.01 |
| 1024 | 512 | 0.5101 | 0.4577 | 1.11 |
| 4096 | 64 | 0.4635 | 0.4691 | 0.99 |
| 4096 | 128 | 0.8434 | 0.8337 | 1.01 |
| 4096 | 256 | 1.7751 | 1.7713 | 1.00 |
| 4096 | 512 | 6.8406 | 6.3971 | 1.07 |

Ratios above 1 favor CuTe; below 1 favor native. This small grid is near parity
(about 0.98–1.11x) and does not establish a general speedup or an improvement over
the previous vendor snapshot. These eager attention-call measurements include
launch overhead and are not image/video generation timings.

[Machine-readable evidence](vendor-sync-2026-11-02.json) includes all timing
samples, runtime versions, hardware, source hash, and validation summaries.
Raw captures remain outside the tracked tree.

Only RTX 4090 kernels were GPU-tested in this integration. Other supported GPUs
and explicit experimental entry points were not GPU-qualified here. Full
image/video generation with pretrained checkpoints was not tested. Historical
records retain their original revisions and environments.

Reproduce from the repository root in the corresponding environment:

```sh
python -m unittest discover -v
python -m tests.gpu_validation --comfyui-root ../ComfyUI
python -m tests.comfyui_smoke --comfyui-root ../ComfyUI
python -m tools.benchmark_attention
```
