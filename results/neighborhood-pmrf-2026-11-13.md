# Neighborhood attention and PMRF validation — 2026-11-13

The new **CuTe Neighborhood Attention (PMRF)** node passed a real PMRF
restoration and ComfyUI HTTP queue test on a rented Vast.ai RTX 4090. It uses
the vendored public `neighborhood_attention` API; it does not patch Torch SDPA
or the installed NATTEN package. Raw logs, images, source snapshots, and launch
scripts were retained outside the tracked tree.

## Sources and environment

- KohakuFA-CuTe: `c6f72ca8d9c9e9cb053cce3722e5f6f2f7aa0704`.
- ComfyUI: `b894a864de9c0c73cc0ca848acd660fca890cebc`.
- [ComfyUI-PMRF](https://github.com/2kpr/ComfyUI-PMRF):
  `e105c042161785b6fbbae840e70817896c1eebf4`.
- [PMRF blind face restoration weights](https://huggingface.co/ohayonguy/PMRF_blind_face_image_restoration):
  `67e0aa382ab4ddc56fdbb576ca8ca5928768113c`.
- Input: PMRF demo's
  [00000055.png](https://huggingface.co/spaces/ohayonguy/PMRF/blob/main/examples/00000055.png),
  SHA256 `68b6a6e507852f97a257b377f0621a6fb692666f5afe2d2c579d151600c34f9e`.
- NVIDIA GeForce RTX 4090, SM89, 128 SMs, 24 GB VRAM, 400 W power limit,
  driver 610.57.04; Ubuntu 24.04 container.
- Python 3.13.18, Torch 2.13.0+cu130, torchvision 0.28.0+cu130,
  CUTLASS DSL 4.8.1 with CUDA 13 libraries, TVM FFI 0.1.15.post0,
  Triton 3.7.1, CUDA bindings 13.0.3, NATTEN 0.21.7+torch2130cu130.
- BasicSR 1.4.2, facexlib 0.3.0, Real-ESRGAN 0.3.0,
  PyTorch Lightning 2.6.6, huggingface-hub 2.4.0.

The neighborhood comparison stack is within this project's declared requirements
and matches upstream's neighborhood qualification stack. The optional dense
FlashAttention-derived runtime requires different pins and was not installed.

## Integration and numerical checks

All 58 unit tests passed on macOS (Python 3.13.18/Torch 2.15.0) and on the Linux
GPU host. This includes all 20 privacy-guard tests. Source distribution and wheel
builds passed; all 60 vendored package files are included.

Publication follow-up: the privacy guard now parses Git tree filenames separately
from binary object IDs, preventing hash bytes from causing false positives.
All 22 guard tests passed locally after adding filename-history and binary-tree
regressions. The GPU campaign counts above describe the original test snapshot.

The CUDA neighborhood suite passed FP16/BF16 forward and backward comparisons
against independently enumerated FP64 shifted windows, including transposed
spatial views, rectangular/even windows, boundary tails, and positive, negative,
and zero scales. The largest component relative L2 error across these fixtures
was 0.03214% FP16 and 0.25378% BF16, below the declared 0.5% and 1% gates.
These are whole-component norms for these fixtures, not universal error bounds.

The NATTEN comparison uses identical rounded `[1, 32, 40, 3, 64]` operands,
window 7, scale 1, and explicit `cutlass-fna`. CuTe/NATTEN output relative L2
differences were 0.00702% FP16 and 0.05612% BF16. Unsupported dtype, head dimension,
and geometry checks passed. A separate GPU test verified explicit FP32-to-FP16/BF16
operand conversion and exact restoration of the FP32 activation output contract.

The existing dense GPU suite passed seven groups, including actual ComfyUI
ModelPatcher integration, masked/GQA attention, FP64 output/gradient checks,
boundary cases, and CUDA graph replay. Two optional-runtime groups were skipped.
The real ComfyUI loader test also passed process override, disable, and model scope.
The dense regression is relevant because the imported neighborhood implementation
extends shared kernel traversal code.

## Full restoration

The example workflow is
`LoadImage → CuTeNeighborhoodAttention → SaveImage`.
The comparison ran the actual ComfyUI node implementations and loaded weights,
using one 512×512 RGB image, seed 123, 25 flow steps, scale 1, Lanczos4, and BF16
neighborhood operands. All other PMRF computations retained their existing dtype.
The NATTEN control uses its default backend dispatch with the same explicit
operand rounding as CuTe; this is not a comparison against original FP32 attention.

Every run executed all eight HDiT neighborhood blocks on every flow step:
**200 neighborhood calls per restoration**. Observed NATTEN-layout Q/K/V shapes:

| Shape | Window | Attention scale | Input / attention dtype |
| --- | --- | --- | --- |
| `[1, 128, 128, 4, 64]` | 7×7 | 1 | FP32 / BF16 |
| `[1, 64, 64, 8, 64]` | 7×7 | 1 | FP32 / BF16 |

The temporary instance forwards were restored after every run. Both backends
returned finite `[1, 512, 512, 3]` images. The output comparison passed the
predeclared gate of mean absolute error ≤2/255 and PSNR ≥35 dB:

- Mean absolute difference: **0.00005706** on the 0–1 output range.
- Maximum absolute difference: **2/255**, within floating-point representation.
- PSNR: **66.38 dB**.

The HTTP queue test separately started the real ComfyUI server, discovered the
node through `object_info`, submitted a two-step prompt, executed **16 CuTe calls**,
and verified a successful history entry and saved image.

## Timings and limits

Timings bracket the complete restoration node with CUDA synchronization and a
wall clock. They include face detection/alignment, the SwinIR source model,
flow steps, attention casts/layout work, restoration compositing, and upsampling.
They exclude initial PMRF weight loading, LoadImage, SaveImage, and queue overhead.
The first invocation includes compilation and other first-use initialization;
the first NATTEN invocation also downloaded face-helper weights.

| Backend | First invocation | Three warm samples | Warm median |
| --- | ---: | --- | ---: |
| NATTEN | 12.520 s | 1.306, 1.284, 1.329 s | 1.306 s |
| CuTe | 10.428 s | 1.235, 1.222, 1.294 s | 1.235 s |

The warm median is about 5.4% lower for CuTe in this small sample. This is an
end-to-end restoration-node observation, not an attention-kernel benchmark or a
general speedup claim. One portrait and seed do not establish perceptual quality
across a dataset. This integration was tested on the RTX 4090 only.
Full timings and environment details are in the [sanitized JSON](neighborhood-pmrf-2026-11-13.json).

## Reproduction notes

Install ComfyUI and ComfyUI-PMRF at the revisions above with the listed runtime,
PMRF weights, and RealESRGAN_x2plus weights. Use the
[adapted workflow](../workflows/pmrf-neighborhood.json) for interactive execution.

BasicSR 1.4.2 needs two compatibility fixes in this Python 3.13 environment:
its packaging `get_version()` must execute the version file in an explicit
dictionary and read `namespace['__version__']`, rather than relying on the changed
semantics of `locals()`. Its `data/degradations.py` must import
`rgb_to_grayscale` from `torchvision.transforms.functional`, the same import fix
already performed by ComfyUI-PMRF's installer. These changes affect packaging
and imports; the PMRF/HDiT model source and weights were not changed. Setuptools
78.1.0 supplies `pkg_resources` for the older PMRF startup script.

From this repository, using ComfyUI's environment and a capture directory outside
the tracked tree:

```sh
python -m unittest discover -v
python -m tests.gpu_neighborhood -v
python -m tests.gpu_validation --comfyui-root ../ComfyUI
python -m tests.comfyui_smoke --comfyui-root ../ComfyUI
python -m tests.pmrf_smoke --comfyui-root ../ComfyUI \
  --input pmrf-example.png --output ../pmrf-captures --steps 25 --repeats 3
```

Place the chosen image in ComfyUI's input directory before the last command.
The smoke test records actual call counts and shapes and fails if no face
restoration occurs, outputs are nonfinite, temporary forwards leak, or output
agreement exceeds its gate. Raw captures must be reviewed before publication.
