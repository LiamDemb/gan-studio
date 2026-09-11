# Source provenance and commercial-use scope

Recorded 10 September 2026.

This project was written in this session from the published mathematical
description and general PyTorch/Triton APIs. No NVIDIA StyleGAN implementation
source, kernels, checkpoints, pickles, utilities, data conversion scripts or
third-party StyleGAN implementation source was downloaded, imported, translated
or used as a coding reference. No external GAN package is a dependency.

The research process retrieved the paper, its appendix, NVIDIA's licence text,
PyTorch documentation, hardware specifications and IP Australia material.
Search results also surfaced repository descriptions and links. Those snippets
were not used as implementation templates. NVIDIA repository implementation
files were not opened. This is an account of this session's work, not a formal
two-team clean-room certification or a claim about the model's pretraining data.

The new project source is offered under the MIT licence in `LICENSE`, which
allows commercial use. That licence does not relicense dependencies, datasets,
weights or third-party patents, and it does not certify exclusive ownership of
an AI-assisted work.

NVIDIA's original implementation is **source available**. Its published licence
limits use of that work and derivatives to non-commercial research or
evaluation. Reading that licence is different from using its implementation.
See [NVIDIA's licence, section 3.3](https://nvlabs.github.io/stylegan2/license.html).

Independent source provenance is useful, but it is not a freedom-to-operate
opinion. Computer-related methods can potentially be patented in Australia;
see [IP Australia's guidance](https://www.ipaustralia.gov.au/patents/what-are-patents/what-computer-related-inventions-can-be-patented).
No claim-by-claim patent search, territorial analysis or legal clearance has
been performed. Obtain that assessment for the intended commercial deployment
before describing the product as legally cleared.

## Dependency boundary

| Component | Use | Treatment |
| --- | --- | --- |
| PyTorch | Autograd, convolution dispatch, optimisers, compiler | Installed dependency; its own licence and bundled notices apply |
| NumPy | Prepared data, feature statistics | Installed dependency; its own licence applies |
| Pillow | One-time image preprocessing, PNG export | Installed dependency; its own licence applies |
| Triton, optional | Our FIR kernels, compiler-generated GPU code | Installed dependency; its own licence applies |
| CUDA, cuDNN, NVIDIA driver | General GPU runtime used through PyTorch | Their separate vendor terms apply |
| pytest, optional | Tests | Development dependency |
| Training images | Supplied by the user | Not distributed; user chooses a dataset with appropriate rights |
| Feature extractor / weights | Supplied externally for quality evaluation | Not bundled or downloaded |

The instruction to avoid NVIDIA code is interpreted as avoiding NVIDIA's
StyleGAN implementation, while allowing the ordinary GPU driver, CUDA and
cuDNN runtime needed to target NVIDIA hardware. A prohibition on every NVIDIA
runtime component would require a materially different project.

## Contribution rule

Do not paste or port code from NVIDIA StyleGAN repositories or wrappers that
carry it. An MIT-labelled repository is not sufficient evidence that every
vendored file is commercially reusable. Record sources for new algorithms,
derive kernels from equations, and test them against this project's own
mathematical reference. Keep any external baseline measurements as data only.

`SOURCE_MANIFEST.json` records SHA-256 hashes of the delivered project files
(excluding itself). It identifies this delivery; hashes are not proof of
copyright provenance or legal clearance.
