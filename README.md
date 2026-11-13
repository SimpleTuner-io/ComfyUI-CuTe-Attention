# ComfyUI CuTe Attention

ComfyUI nodes using a vendored copy of
[KohakuFA-CuTe](https://github.com/SimpleTuner-io/KohakuFA-CuTe):

- **CuTe Attention (KohakuFA)** replaces dense attention on a `MODEL`.
- **CuTe Neighborhood Attention (PMRF)** restores an `IMAGE` using PMRF's
  neighborhood-attention layers and returns an `IMAGE`.

Importing the extension does not enable an override or load PMRF weights.

## Install

Place this directory under ComfyUI's `custom_nodes`, then run the following from
this directory with ComfyUI's Python environment:

```sh
python -m pip install -r requirements.txt
```

Restart ComfyUI. Requires Python 3.12+, CUDA-enabled Torch 2.11+, Linux, an NVIDIA
GPU supported by the vendored kernels, and a C++17 compiler for CuTe execution.
The native launcher and kernels compile on first use, which can take a while.
The vendored source is KohakuFA-CuTe 0.4.0 at
`c6f72ca8d9c9e9cb053cce3722e5f6f2f7aa0704`; see [provenance](VENDORED.md).
It is included here and is never downloaded at runtime.
CPU and other devices use native Torch SDPA in `auto` mode.

For a CUDA 13 Torch environment, also install the matching DSL libraries:

```sh
python -m pip install 'nvidia-cutlass-dsl[cu13]>=4.8,<4.9'
```

## PMRF neighborhood attention

Install [ComfyUI-PMRF](https://github.com/2kpr/ComfyUI-PMRF), its dependencies,
and the [PMRF face-restoration weights](https://huggingface.co/ohayonguy/PMRF_blind_face_image_restoration).
Its original [workflow](https://github.com/2kpr/ComfyUI-PMRF/blob/e105c042161785b6fbbae840e70817896c1eebf4/ComfyUI-PMRF-Workflow.json)
uses NATTEN. Load our [adapted workflow](workflows/pmrf-neighborhood.json), choose
an input face image, and queue it:

`Load Image → CuTe Neighborhood Attention (PMRF) → Save Image`

The node uses the already loaded PMRF model. `scale`, `num_steps`, `seed`, and
`interpolation` have the same meaning as in the original PMRF node.
`precision=bfloat16` (default) or `float16` explicitly rounds only neighborhood
Q/K/V after PMRF's cosine scaling and rotary embedding. Attention output returns
to the original activation dtype before PMRF's output projection. This differs
from full FP32 attention; it does not convert all model weights or enable global
autocast. `backend=cute` is strict: unsupported inputs or kernel failures raise.
`backend=natten` is an optional comparison on identically rounded operands and
requires a NATTEN build compatible with the installed Torch/CUDA versions.

The CuTe path itself does not import NATTEN. The third-party PMRF extension's
startup installer may still attempt to install it; its old automatic installer
targets older Torch/CUDA combinations. Use a supported ComfyUI environment and
install PMRF dependencies deliberately. This adapter targets the unmodified
HDiT neighborhood blocks in ComfyUI-PMRF revision
`e105c042161785b6fbbae840e70817896c1eebf4`. Existing forward overrides and compiled
PMRF models are unsupported. The selected backend applies only during this node's
execution, and temporary instance forwards are restored on success or failure.
Other threads retain the original forward. Torch SDPA and the NATTEN package are
not patched. A face that is not detected may produce no neighborhood calls;
the node logs a warning in that case.

The underlying API supports matching FP16/BF16 CUDA tensors with head dimension
64, row-major 2D geometry, stride/dilation one, and shifted boundary windows.
It does not support attention masks, dropout, causal windows, relative positional
bias, GQA, or other head dimensions. It preserves upstream's corrected neighborhood
gradient policy. Neighborhood attention is separate from the dense SDPA node;
enabling the dense node does not replace NATTEN calls.

See the [RTX 4090 PMRF validation](results/neighborhood-pmrf-2026-11-13.md)
for the tested environment, dependency compatibility fixes, output comparison,
and complete restoration timings.

## Dense attention

Connect the node after your model loader and model patches, and connect its
`MODEL` output to the sampler. Inputs:

| Input | Behavior |
| --- | --- |
| `enabled` | Enables the selected override; defaults to true. |
| `scope=process` | Default. Routes subsequent calls to `torch.nn.functional.scaled_dot_product_attention` throughout ComfyUI through CuTe selection. Also overrides the connected model's optimized attention backend. |
| `scope=model` | Overrides the cloned model's optimized attention backend and direct Torch SDPA calls within its `apply_model` execution. Other execution contexts keep the process setting. |
| `mode=auto` | Uses CuTe for supported calls and native Torch SDPA for unsupported inputs or missing CUDA dependencies. |
| `mode=strict` | Requires supported CuTe inputs and dependencies. Unsupported calls and kernel errors propagate. |

**Process scope is a persistent process-wide setting.** It starts when the node
executes, so text encoding or other work that ran earlier is unaffected. Removing
or bypassing the node does not reset it. Run the node with `enabled=false` and
`scope=process`, or restart ComfyUI, to clear the process setting. If multiple
process nodes run, the last one executed sets the process default. The node
re-executes on each queued prompt so cached node outputs do not skip this action.
Use model scope when different workflow branches need different settings.

Disabling the process setting does not remove overrides from previously patched
`MODEL` outputs. Use the original model output to stop that model override.
The lightweight Torch dispatcher remains installed after disabling the process
setting; outside an active patched model it calls the original SDPA. Model scope
uses execution-context state, restored even when the model raises. It does not
change an already enabled process setting. Existing model patches on the input
are preserved by cloning, but the optimized attention backend override is
replaced. Place backend-selection nodes deliberately when combining extensions.

Process scope affects Python calls that resolve Torch SDPA after installation.
Previously captured function aliases, already compiled graphs, direct ATen/C++
calls, and independent xFormers/FlashAttention calls outside the patched model
are outside that interception. Existing compiled graphs need rebuilding after
changing attention settings. The execution-context dispatcher can cause graph
breaks; `torch.compile(fullgraph=True)` through that dispatcher is unsupported.
The vendored attention functions themselves retain upstream compiler support.

## Supported calls

CuTe supports matching FP16/BF16 CUDA tensors in `[B, H, S, D]` layout, head
dimensions 1–512, GQA, causal attention, and broadcastable boolean masks (`True`
means visible). Supported compute capabilities are SM80, SM86, SM89, SM90, SM100,
SM103, SM120, and SM121. GQA requires `enable_gqa=True`. Explicit masks combined
with `is_causal=True`, floating-point additive masks, dropout, FP32 inputs,
unsupported shapes/devices, and missing CUDA dependencies use native SDPA in
`auto` mode. Errors from a selected CuTe kernel are never silently swallowed.

The ComfyUI adapter accepts flattened `[B, S, H*D]` or pre-shaped `[B, H, S, D]`
tensors, ComfyUI's 2D/3D/4D mask conventions, custom scales, GQA, and either output
layout. It follows ComfyUI's PyTorch attention behavior for `attn_precision`.
All-hidden boolean-mask rows return zero through the CuTe backend.

Selecting CuTe does not guarantee a speedup. Benchmark your actual model and
shapes, including first-use compilation separately from warm inference.
See the [RTX 4090 validation and timings](results/README.md) for measured results
and the tested ComfyUI environment.
The [previous vendor refresh validation](results/vendor-sync-2026-11-02.md)
records the integration checks and hardware coverage for that earlier snapshot.

## Optional tuned runtime

The normal install keeps the supported Torch 2.11+ environment. Some upstream
0.4.0 schedules require a separate, exactly qualified runtime: **Python 3.13,
Torch 2.15.0 built for CUDA 13.2**, CUTLASS DSL 4.8.1, TVM FFI 0.1.15.post0,
quack-kernels 0.6.5, torch-c-dlpack-ext 0.1.5, and a pinned FlashAttention 4
source revision. With that Python/Torch environment already installed, run:

```sh
python -m pip install '.[cuda,sm80-fa4]'
python -m pip install 'nvidia-cutlass-dsl[cu13]==4.8.1'
```

This optional install needs Git and may replace packages in ComfyUI's environment;
check compatibility with your other nodes first. A different Python, CUDA build,
package version, or FlashAttention source fingerprint leaves these optional paths
unavailable. Restart after changing dependencies. The ordinary CuTe backend
remains available under its base requirements.

Both node modes inherit upstream's automatic selection. On a full RTX 4090,
the new short and shared schedules target contiguous, aligned B1/H16 training
calls at S128/D16 and S1024 or S4096 with D64/D128, with additional policy
checks. **Those automatic RTX 4090 paths require gradients; ordinary ComfyUI
inference keeps the ordinary CuTe path.** Upstream's training speedups therefore
do not establish faster image or video generation. See the pinned
[automatic selection notes](https://github.com/SimpleTuner-io/KohakuFA-CuTe/blob/30704427809008e83ad38174e7c1780016c44258/docs/rtx4090-automatic.md)
and [S1024 update](https://github.com/SimpleTuner-io/KohakuFA-CuTe/blob/30704427809008e83ad38174e7c1780016c44258/docs/rtx4090-s1024-host.md).

Other automatic paths have their own hardware and input guards. Upstream also
includes explicit experimental entry points, including Hopper/H200 and selected
A40, RTX 3090, and GB10 schedules. Installing an extra does not make the node
select every explicit upstream entry point; the dense node exposes only upstream's
public automatic and strict SDPA APIs. The `hopper-fa4` extra is retained for
upstream API experiments and is unnecessary for normal node use.

## Precision and determinism

Version 0.4.0 changes the default backward arithmetic. Probabilities and score
gradients are rounded once to the FP16/BF16 input dtype, with FP32 accumulation.
The backward normalization uses the actual rounded forward output. Earlier
compensated/residual products are no longer enabled by default. Small terms may
underflow and cancellation-sensitive gradients can differ from both the previous
vendor snapshot and FP64 mathematical attention. The node adds no compensation
or key-centering option. See upstream's
[arithmetic contract](https://github.com/SimpleTuner-io/KohakuFA-CuTe/blob/30704427809008e83ad38174e7c1780016c44258/README.md).

`mode=strict` controls backend fallback. For deterministic training, separately
set `torch.use_deterministic_algorithms(True)` before forward execution, warm-up,
or graph capture. This excludes optional unordered atomic-gradient schedules.
Enabling it after a forward selected such a schedule makes backward raise;
`warn_only=True` warns and permits execution. CUDA graph replay does not recheck
host policy, so recapture graphs after changing it.

## Development

```sh
python -m unittest discover -v
python -m tests.gpu_validation
# With a local ComfyUI checkout, also check the actual loader and patcher:
python -m tests.gpu_validation --comfyui-root ../ComfyUI
python -m tests.comfyui_smoke --comfyui-root ../ComfyUI
python -m tests.gpu_neighborhood
python -m tools.benchmark_attention
```

The GPU validation command requires a supported CUDA GPU and the CUDA runtime
dependencies. It tests real CuTe dispatch, layouts, masks, GQA, gradients, and
native fallback. Boundary and graph checks use the rounded-operand reference;
FP64 comparisons remain a separate mathematical accuracy check. Unit tests
exercise scope isolation, restoration on errors, and deterministic policy.
See `AGENTS.md` for the privacy and publication workflow, and `VENDORED.md` for
source provenance and local integration changes.

## License

AGPL-3.0-or-later. See `LICENSE`, `NOTICE`, and `LICENSES/` for retained
NVIDIA CUTLASS and FlashAttention BSD notices and author attribution. Optional
FlashAttention dependencies retain their own licenses.
