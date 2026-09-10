# Independent StyleGAN2: design, evidence and performance roadmap

Target: one RTX 4090, 512 × 512 to 1024 × 1024 images.

## What the research supports

[Karras et al., StyleGAN2, including Appendix B](https://arxiv.org/html/1912.04958v2)
is the algorithm specification. Its key changes are weight demodulation,
path-length regularisation and fixed-resolution training with a skip generator
and residual discriminator. The appendix describes lazy regularisation and
the optimiser correction that accompanies its extra updates. It also identifies
filtering and activation overhead as worthwhile optimisation targets. This
project implements those ideas independently, with its own operation contracts
and tests. It does not claim checkpoint or pixel-level compatibility with the
authors' software.

[PyTorch's performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
supports channels-last convolution layouts, mixed precision, compiler fusion,
pinned data loading and avoiding unnecessary synchronisation. These are design
inputs, not evidence of a particular speedup for this network.

[PyTorch's double-backward issue](https://github.com/pytorch/pytorch/issues/91469)
and [AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
motivate a conservative split: compiled, mixed-precision ordinary loss paths;
eager, FP32 regulariser paths. Both the forward pass used to construct a
regulariser and its derivatives stay outside compilation. Merely moving the
second backward call outside a compiled function is insufficient.

[Custom operator guidance](https://docs.pytorch.org/tutorials/advanced/python_custom_ops_registrations.html)
requires an explicit autograd registration. The FIR adjoint is itself a custom
operator with an autograd rule, enabling higher derivatives through the filter.

[NVIDIA lists 24 GB of memory for the RTX 4090](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/).
The configurations here use BF16 and gradient accumulation to target that
budget. Actual peak memory must be measured; these are starting profiles.

## Implementation contract

| Component | This implementation |
| --- | --- |
| Latents | 512-dimensional Z and W by default |
| Mapping | Eight equalised fully connected layers, 0.01 learning-rate multiplier |
| Synthesis | Learned 4² constant, modulated convolutions, per-layer noise, RGB skip accumulation in FP32 |
| Style slots | `2*log2(resolution)-2`; an intermediate RGB branch shares a W slot with the next scale's first convolution |
| Discriminator | Residual downsampling, minibatch deviation feature, scalar logits |
| Channels | `min(32768/resolution, 512)` by default, independently configurable |
| Adversarial objective | Non-saturating logistic generator and discriminator losses |
| Regularisation | Lazy R1 and path length, separate optimiser steps, shared Adam state per network |
| Defaults | R1 every 16 main steps, PL every 8, weights 10 and 2 respectively |
| Stability | FP32 weights, mapping, style affines and norms; FP32 regularisers; FP32 RGB sum |
| State | Generator EMA, W average, PL target, Adam moments, scalers, counters and per-rank RNG |
| Inputs | Square power-of-two RGB images, stored as uint8 NHWC arrays |

The implementation is unconditional StyleGAN2. It has no ADA, class conditioning,
progressive growing, projection encoder, imported weights or NVIDIA pickle
loader. Limited-data quality needs a separately validated augmentation design.

### Our modulation factorisation

Let K have shape `[O,I,k,k]`, S have shape `[N,I]`, and X have shape
`[N,I,H,W]`. K already includes its equalised scaling. The direct reference
forms one effective kernel per image. The alternative used here computes:

```
A[o,i] = sum_uv K[o,i,u,v]^2
D[n,o] = rsqrt(sum_i S[n,i]^2 A[o,i] + epsilon)
Y = conv2d(X * S[...,None,None], K) * D[...,None,None]
```

Scaling an input channel before a linear convolution is equivalent to scaling
that channel's weights. Output-channel scaling then applies demodulation. This
derivation allows an ordinary batched convolution, instead of groups equal to
the batch size. It also replaces a batch-expanded weight-norm reduction with
a small matrix multiplication. The output RGB layers omit D.

For N=4, I=O=512 and k=3, a batch-expanded FP32 kernel alone contains 9,437,184
elements, about 36 MiB. The unexpanded kernel contains about 9 MiB. These are
allocation calculations, not measured training-memory savings: autograd saved
tensors, casts and activation traffic change the total. At high resolution,
activation modulation may cost more than the saved weight traffic. Both
`factorized` and `grouped` are therefore benchmarkable. No universal winner is
hard-coded as a measured fact.

Mixed/unmixed style tensors are made contiguous before synthesis to keep
compiler input strides stable. FP32 norm computation avoids reducing the
demodulation denominator in BF16. TF32 is an explicit opt-in, rather than an
unrecorded numerical change.

### Our resampling contract

The FIR uses the outer product of `[1,3,3,1]`, normalised by 64, and gain `up²`.
An input is zero-inserted with trailing zeros, padded, correlated and decimated.
For each spatial axis:

```
output_size = (input_size*up + left + right - 4) // down + 1
source_coordinate = (output_coordinate*down + tap_coordinate - left) / up
```

Only integral, in-bounds source coordinates contribute. Upsampling uses
`(up,down,left,right)=(2,1,2,1)`; downsampling uses `(1,2,1,1)`;
same-size blur uses `(1,1,1,2)`. Padding and sampling origin are part of the
contract and are tested, including odd input sizes and non-contiguous layouts.

The PyTorch reference explicitly performs these operations. The experimental
Triton forward gathers the 16 possible contributions directly, avoiding the
zero-inserted tensor, padded tensor and separate depthwise convolution.
Its gather adjoint inverts the coordinate equation and avoids atomics.
It returns channels-last storage for the convolution pipeline.

The filter is linear, so if its matrix is A, forward is `A*x`, backward is
`A.T*v`, and backward through that adjoint is `A*u`. The custom operator
registrations implement this relationship. It is essential for R1 and PL.
Triton's interpreter checks the maths; it cannot establish device correctness,
coalescing performance, launch behaviour or mixed-precision accuracy.

The generator filters an upsampled feature map before its spatial convolution.
The discriminator blurs before its stride-2 convolution. Boundary conventions
are explicit and independent. They are not asserted to match NVIDIA's fused
convolution/filter ordering at every border pixel.

### Regularisation and training state

R1 differentiates the sum of discriminator logits with respect to real images,
then penalises the mean squared image-gradient norm. PL differentiates a
noise-weighted image sum with respect to W slots, computes the RMS norm over
slots and penalises deviation from a running target.

Image noise is scaled by `1/sqrt(H*W)` and the slot squared norms are averaged.
With `L=2*log2(R)-2`, our default PL weight 2 corresponds to an unnormalised
squared-norm coefficient `2/(R²*L)`. This is an algebraic explanation of the
normalisation, not an empirical tuning result.

For a regularisation interval k, the regulariser's loss weight is multiplied
by k. The associated Adam learning rate is scaled by `k/(k+1)` and its betas
are raised to that power. The correction is disabled when the regulariser
weight is zero. A regulariser runs after k completed main updates, so the
phase counter resumes without an off-by-one change.

Gradient accumulation weights each equal-sized microbatch by its share of the
effective batch. PL uses the previous target for every microbatch, and updates
the target once using all samples/ranks. This ordering deliberately makes the
target independent of the microbatch partition; it can differ from another
implementation's target-update ordering. W averages update once per generator
main phase. EMA ramp-up and precision choices are configurable engineering
choices, not a claim of exact reproduction of every paper experiment.

Minibatch deviation is local to a microbatch. At 1024², the supplied microbatch
of 2 uses a deviation group of 2. Accumulating to 32 does not restore group-4
statistics. If memory permits, compare microbatch 4 while holding the effective
batch fixed. Treat that as a training change, not just a memory setting.

Distributed support reduces gradients after each phase using fixed-order
buckets and explicitly synchronises averages and EMA inputs. It avoids DDP's
interaction with `autograd.grad`. It does not overlap communication with
backward; single-card 4090 performance is the current priority.

## Performance acceptance criteria

“Rivals NVIDIA” remains an unverified goal. A suggested acceptance threshold is
at least 90% of an agreed baseline's effective image throughput, while staying
within 24 GB and producing comparable quality across multiple seeds. The 90%
threshold is a proposed project criterion, not a benchmark result.

For a fair comparison, record the same GPU and power limit, driver/runtime,
resolution, channel schedule, effective batch, actual deviation-group size,
regulariser weights/intervals, style mixing, augmentation and optimiser policy.
Record precision differences explicitly. Distinguish the original TensorFlow
StyleGAN2 baseline from the later PyTorch StyleGAN2-ADA baseline. “NVIDIA speed”
without a version and training configuration is not a reproducible target.

Measure:

1. Full-cycle steady-state images/s and seconds/kimg, including D, G, R1, PL,
   optimiser steps and EMA. Do not report generator-only speed as training speed.
2. Peak allocated and reserved memory, compile/warm-up time, and per-cycle step
   latency. Reject steady-state results that include fresh compilation.
3. Real-data wall-clock throughput including preprocessing/loading, logging,
   samples and checkpoints. The benchmark command is separately labelled as
   device-resident synthetic input.
4. Quality versus both images seen and elapsed time, with an agreed Inception
   feature extractor/preprocessing, fixed evaluation sample count and seed set.
   Do not assume feature distances from different pipelines are comparable FID.

No NVIDIA evaluation code or weights are required by this repository. An
external baseline can supply measurement data, with its provenance recorded
separately; no implementation source needs to enter this project.

## Next hardware experiments, in order

| Gate | Experiment | Decision |
| --- | --- | --- |
| CUDA correctness | `tests/test_cuda.py` | All fused output, gradient, higher-derivative and custom-op checks pass |
| 512² baseline | Eager BF16 factorized/torch, batch 32, microbatch 4 | Establish memory and full-cycle speed |
| Compiler | Enable main-path compilation only | Keep if steady-state throughput improves after startup is accounted for |
| FIR | Enable independent Triton filter | Keep only if correctness gates and full-cycle speed both improve |
| Modulation | Compare grouped and factorized under identical settings | Select using whole-model speed and memory, not intuition |
| 1024² | Repeat with microbatch 2, then try 4 | Determine what fits and quantify deviation-group/quality effects |
| Quality | Train real data at both resolutions, several seeds | Assess convergence and time-to-quality before a parity claim |

After profiling, further candidates include shape-specific modulation dispatch,
separable or tiled FIR kernels, fusing filtering with neighbouring convolution,
optimised convolution higher derivatives, a fused bias/noise/activation path
that preserves higher derivatives, and overlapping host-to-device transfer.
These are pending experiments, not implemented features. Do not replace cuDNN
with a hand-written convolution solely to make the code look more optimised.
