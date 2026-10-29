# RTX 4090 validation

Validated on 2026-10-28 against ComfyUI **0.43.0**, revision
`35060f0156181ae78a86b10f098f67ce8e83697f`, using KohakuFA-CuTe revision
`736c23f89a4bdf940a421f76c6909fe60c4c4142`.

The 2026-10-29 vendor refresh to `a30bc8702ef58d6e3a7336d9fc623fcfb0801785`
adds GB10/SM121 optimizations. The results below remain measurements of the
earlier revision. The refresh passed the 32 CPU unit tests, syntax checks, wheel
packaging checks, and the repository privacy scan; its new GB10 kernels have not
been validated on GPU in this integration.

The GPU environment used Python 3.12.3, Torch 2.11.0, driver 580.82.07, CUTLASS
DSL 4.8.1, TVM FFI 0.1.15.post1, Triton 3.6.0, and Ninja 1.13.0. Validation ran
on both CUDA 12.8 and CUDA 13.0 Torch builds. Timings below use CUDA 13.0 and the
matching DSL libraries.

## Correctness and integration

- All 32 unit tests passed on macOS (Python 3.13, Torch 2.15) and on the Linux
  GPU host (Python 3.12, Torch 2.11/CUDA 12.8). These include the privacy guard,
  layout adapter, native fallback, scope isolation, and exception restoration.
- All five GPU validation groups passed on each CUDA build. Strict CuTe calls
  exercised FP16/BF16, head dimensions 64/128/256/512, strided inputs,
  unequal query/key lengths, causal attention, boolean masks, all-hidden rows,
  GQA, and negative scales. Outputs and gradients were compared with native
  Torch SDPA in FP64 on identical rounded inputs.
- The largest observed gradient L2 error divided by reference gradient L2 norm
  was below **0.03% for FP16** and **0.24% for BF16** across the ten recorded
  numerical cases. These bounds describe these inputs, not every workload.
- Both builds passed the real ComfyUI custom-node loader smoke test, including
  the native launcher's import under ComfyUI's module name, the optimized
  attention hook, direct Torch calls, process disable, and model-scoped wrappers.
- Direct vendored attention passed full-graph Dynamo capture with the `eager`
  backend. This does not establish full-graph compilation through the scoped
  Torch dispatcher; that limitation is described in the main README.

These checks cover attention and integration hooks. Full image or video
generation with pretrained checkpoints has not been validated. Other GPU
architectures retain upstream support but were not tested in this integration.

## Warm forward timings

Batch 1, 16 heads, BF16, dense self-attention with independent random Q/K/V;
BHSD views of BSHD storage. CUDA events measured 30 calls per repeat and five
repeats; the table reports the median of those repeated means. Compilation and
warm-up are excluded. Measurements include the process SDPA dispatcher on the
CuTe side and use Torch's default backend selection for the native side.

| Sequence | Head dim | Native SDPA (ms) | CuTe override (ms) | Native / CuTe |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 0.0346 | 0.0382 | 0.906 |
| 1024 | 128 | 0.0658 | 0.0722 | 0.911 |
| 1024 | 256 | 0.1241 | 0.1361 | 0.912 |
| 1024 | 512 | 0.4633 | 0.4371 | 1.060 |
| 4096 | 64 | 0.4311 | 0.4748 | 0.908 |
| 4096 | 128 | 0.8617 | 0.9939 | 0.867 |
| 4096 | 256 | 1.8202 | 1.9766 | 0.921 |
| 4096 | 512 | 6.8274 | 6.4334 | 1.061 |

A separate profiler pass identified native FlashAttention for dimensions
64/128/256 and native memory-efficient attention for dimension 512. Profiling
was outside the timing loop. CuTe used strict dispatch to ensure these timings
could not silently measure native fallback.

For these shapes, dimension 512 was about 6% faster with CuTe; smaller dimensions
were slower. These are attention-call timings from one machine, and do not
measure end-to-end ComfyUI generation. `auto` selects by compatibility and
dependency availability; it does not select the fastest backend for a shape.

[Machine-readable results](rtx4090.json) retain all timing samples, profiler
operator names, and the per-case numerical errors. Raw machine logs are kept
outside the tracked tree.

Reproduce with the commands in the main README. The timing command is:

```sh
python -m tools.benchmark_attention --seq 1024 4096 --dim 64 128 256 512 \
  --iterations 30 --repeats 5
```
