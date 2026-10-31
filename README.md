# ComfyUI CuTe Attention

A **CuTe Attention (KohakuFA)** node that replaces attention with a vendored copy
of [KohakuFA-CuTe](https://github.com/SimpleTuner-io/KohakuFA-CuTe). It accepts and
returns a `MODEL`. Importing the node does not enable an override.

## Install

Place this directory under ComfyUI's `custom_nodes`, then run the following from
this directory with ComfyUI's Python environment:

```sh
python -m pip install -r requirements.txt
```

Restart ComfyUI. Requires Python 3.12+, CUDA-enabled Torch 2.11+, Linux, an NVIDIA
GPU supported by the vendored kernels, and a C++17 compiler for CuTe execution.
The native launcher and kernels compile on first use, which can take a while.
The KohakuFA-CuTe source is included here; it is never downloaded at runtime.
CPU and other devices use native Torch SDPA in `auto` mode.

For a CUDA 13 Torch environment, also install the matching DSL libraries:

```sh
python -m pip install 'nvidia-cutlass-dsl[cu13]>=4.8,<4.9'
```

## Use

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
means visible). Supported compute capabilities are SM86, SM89, SM90, SM100,
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
The [latest vendor refresh validation](results/vendor-sync-2026-10-31.md)
records the integration checks and hardware coverage for this snapshot.

## Development

```sh
python -m unittest discover -v
python -m tests.gpu_validation
# With a local ComfyUI checkout, also check the actual loader and patcher:
python -m tests.gpu_validation --comfyui-root ../ComfyUI
python -m tests.comfyui_smoke --comfyui-root ../ComfyUI
python -m tools.benchmark_attention
```

The GPU validation command requires a supported CUDA GPU and the CUDA runtime
dependencies. It tests real CuTe dispatch, layouts, masks, GQA, gradients, and
native fallback. Unit tests exercise scope isolation and restoration on errors.
See `AGENTS.md` for the privacy and publication workflow, and `VENDORED.md` for
source provenance and local integration changes.

## License

AGPL-3.0-or-later. See `LICENSE`, `NOTICE`, and `LICENSES/BSD-3-Clause.txt` for
upstream attribution and retained BSD notices.
