# Validation record

10 September 2026.

## gan-studio integration

Checkpoints from this trainer use `latest.pt` with `format_version: 1` and
`G_ema` weights. They are **not** compatible with lucidrains `stylegan2_pytorch`
`model_*.pt` files or its `ModelLoader` API.

## Passed

- **36 CPU tests** in the final pytest suite. These cover modulated-convolution
  outputs and input/weight/style gradients against a literal per-image oracle;
  finite-difference first and second derivatives; FIR coordinates, scaling and
  adjoints; model shapes, style counts, noise and minibatch statistics;
  analytic R1 and PL cases; parameter updates, lazy Adam counts and adjustment;
  exact next-step checkpoint/resume; accumulation gradients; BF16 main passes
  with FP32 regularisation; data preparation, sampler ordering and metric maths.
- **512² and 1024² full-width structural checks** using PyTorch's meta device.
  This checks tensor shapes without allocating activations. It does not test
  high-resolution numerical execution, memory fit or speed.
- **12 Triton CPU-interpreter cases** across resampling parameters and NCHW,
  channels-last and strided inputs. Forward, adjoint and second-derivative
  results agree with the PyTorch reference within the test tolerances.
- **Custom-op schema, autograd-registration and fake-tensor checks** for the
  interpreter test registrations. These CPU registrations are confined to the
  verification process and are not production fallback kernels.
- **Actual CPU Inductor compiled training**, including an R1/PL step through
  the original eager modules. The final recorded measured window contains no
  new compiler graphs. This is a functional smoke test at 16² with a small
  network, not a GPU or statistically robust speed study.
- **Editable package build/install** and single-process image preparation,
  file-backed training, resume and PNG sample commands.

## Not validated

- **11 CUDA tests are skipped** because no CUDA device is exposed. Their test
  cases cover fused FIR output and derivatives in FP32/FP16/BF16, custom-op
  contracts and compiled training with the fused filter.
- Actual Triton CUDA compilation, launch correctness, memory access behaviour
  and mixed-precision accuracy on the RTX 4090.
- 512² or 1024² peak VRAM, CUDA full-cycle throughput, time-to-quality, image
  quality, numerical stability across long runs, or parity with NVIDIA.
- Multi-GPU/NCCL execution. The two-process Gloo validation attempt failed
  before training because this runtime denied TCP device creation.
- Multiworker data loading. Both a fork-based attempt and the subsequent spawn
  attempt stalled in this restricted environment; the timed spawn attempt was
  terminated after 60 seconds. The CLI uses spawn and a configurable worker
  timeout, but this path needs testing in an ordinary training environment.
  `--workers 0` works here and remains the fallback for restricted environments.
- GPU checkpoint bitwise determinism. CPU exactness does not establish this.
- Patent clearance, legal ownership exclusivity or dataset rights.

## Interpreting the timing files

`cpu-smoke-benchmark.json` and `cpu-compiled-smoke.json` contain actual small
CPU training runs. Their warm-up and measurement lengths differ, compilation
caches may be warm, and the machine was also running other verification work.
They demonstrate that the harness executes complete regularised training.
They must not be extrapolated to a 4090 or advertised as a speedup comparison.

The hardware benchmark command includes optimisers and EMA, counts real images
in the effective training batch, synchronises CUDA around the measured window,
and reports whether compilation occurred inside that window. Data is resident
on the device. Use the training CLI for actual data-pipeline throughput.

## Required next run on the target 4090

From a Linux environment with CUDA-enabled PyTorch and compatible Triton:

```bash
python -m pytest -q
bash scripts/benchmark_4090.sh
```

First resolve any failed or skipped GPU gates. Then inspect memory and
`steady_state_valid`, compare the measured configurations, and train real
data. A good result on a tiny network is insufficient to release a fused kernel
as production-validated for full-resolution training.
