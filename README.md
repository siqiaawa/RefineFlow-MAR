# RefineFlow-MAR

Research code for metal artifact reduction (MAR) in dental CT using a
rectified-flow image prior and projection-domain data consistency.

## Release Status

This repository is an initial public snapshot prepared before publication of
the associated paper. It includes the supporting model architecture, solver,
geometry and data-loading modules, evaluation tools, a small synthetic test
set, and example outputs.

The following artifacts are intentionally not included in this snapshot:

- `mar_pnp_rf.py` is a public placeholder; the final inference entry point is
  withheld until a later release.
- `generate_pair_data.py` is a public placeholder; the final synthetic-data
  generation implementation is withheld until a later release.
- The pretrained JiT checkpoint is distributed separately and is not currently
  available from this repository.
- The paper citation will be added after publication.

As a result, this snapshot documents the intended workflow but does not yet
provide a complete end-to-end inference command.

## Method Overview

The intended inference pipeline combines:

1. A single-channel JiT rectified-flow image prior.
2. DPS-style likelihood guidance from the measured sinogram.
3. Differentiable fan-beam projection and backprojection using Torch Radon.
4. Optional post-processing with sinogram replacement, gradient
   data-consistency refinement, or hybrid-mask refinement.

The included default configuration selects `hybrid_mask` refinement. Pixels
inside the estimated metal/artifact region are progressively refined, while
pixels outside that region use one-step sinogram replacement.

## Repository Layout

```text
RefineFlow-MAR/
|-- config/
|   |-- MAR_pnp_rf.yaml
|   `-- MAR_pnp_rf_clinical.yaml
|-- data_dependancy/metal_masks/
|-- geometry/
|   |-- build_gemotry.py
|   `-- syndeeplesion_data.py
|-- model/
|   `-- README.md                       # Checkpoint availability
|-- patch_diffusion/
|   |-- pnp_rf_mar.py                   # DPS-RF solver and refinement
|   `-- jit/                            # JiT model definitions
|-- test_data/
|   |-- gtdata/
|   `-- oral_ct/                        # Included synthetic test data
|-- test_results_dps_rf/                 # Included example outputs
|-- utils/
|-- mar_pnp_rf.py                        # Public-release placeholder
|-- generate_pair_data.py                # Public-release placeholder
|-- inspect_h5_data.py
|-- inspect_h5_hu.py
|-- calculate_psnr_ssim.py
|-- requirements.txt
|-- THIRD_PARTY_NOTICES.md
|-- LICENSE
`-- README.md
```

The directory name `data_dependancy` is retained for compatibility with the
existing project layout.

## Environment

The intended pipeline requires Linux, an NVIDIA GPU, and a working CUDA
toolchain. It uses both Torch Radon and the ODL `astra_cuda` backend.

Recommended prerequisites:

- A Python version supported by the selected PyTorch/CUDA combination.
- CUDA-enabled PyTorch and a matching `torchvision` version.
- CUDA Toolkit (`nvcc`), a C/C++ compiler, and Git.
- The packages listed in [requirements.txt](requirements.txt).

Install PyTorch first using the command appropriate for the server CUDA
version, then install the remaining requirements:

```bash
# Example only: choose the PyTorch command for the target CUDA version.
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

`torch-radon` is installed from a pinned Git commit and builds a CUDA
extension. The dependency is licensed under GPL-3.0; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Most package versions are intentionally not pinned because CUDA servers
require compatible PyTorch, torchvision, compiler, and driver combinations.
For strict reproduction, record the working server environment with
`python -m pip freeze` and the CUDA/PyTorch version information.

## Model Checkpoint

The intended synthetic configuration expects:

```text
model/checkpoint-step0240000.pth
```

The checkpoint is not tracked by Git. Download instructions and a SHA-256
checksum will be added when it is released. See
[model/README.md](model/README.md).

## Included Synthetic Data

The repository contains one synthetic test image combined with ten metal masks:

```text
test_data/oral_ct/
|-- testmask.npy
|-- test_720geo_dir.txt
`-- test_720geo/
    `-- patient_0000/000/
        |-- gt.h5
        |-- 0.h5
        |-- 1.h5
        |-- ...
        `-- 9.h5
```

The H5 datasets use these fields:

- `gt.h5`: artifact-free image stored as `image`.
- `0.h5` through `9.h5`: `ma_CT`, `ma_sinogram`, `LI_CT`,
  `LI_sinogram`, and `metal_trace`.
- `testmask.npy`: image-domain metal masks with shape `H x W x N`.
- `test_720geo_dir.txt`: relative paths to ground-truth H5 files.

The included H5 files use the anonymous identifier `patient_0000` and do not
contain patient or author attributes. Users remain responsible for verifying
the redistribution rights and de-identification status of any replacement
data.

## Intended Inference Configuration

The full public inference entry point is not part of this initial snapshot.
When it is released, configure the following fields in
`config/MAR_pnp_rf.yaml` for the target server:

```yaml
model_path: /path/to/checkpoint-step0240000.pth
data_path: /path/to/test_data/oral_ct
save_dir: ./test_results_dps_rf
```

The intended invocation is:

```bash
CUDA_VISIBLE_DEVICES=0 python mar_pnp_rf.py \
    --config ./config/MAR_pnp_rf.yaml
```

In this snapshot, `mar_pnp_rf.py` contains only a release-status placeholder,
so the command above is documentation for the future complete release.

Important configuration options include:

| Option | Meaning |
| --- | --- |
| `model_name` | JiT model registry key; the default is `JiT-B/16`. |
| `image_size` | Reconstruction image size; included data uses `512`. |
| `num_steps` | Number of rectified-flow integration steps. |
| `sampling_method` | `euler` or `heun`. |
| `grad_mode` | `full` guidance gradient or lower-memory `approx`. |
| `init_mode` | `noise` or `xli_blend`. |
| `final_refine_mode` | `none`, `sino_replace`, `gradient_dc`, or `hybrid_mask`. |
| `hybrid_mask_mode` | `original` or `projection_augmented`. |

## Synthetic Data Generation

The intended generator produces DuDoDp-compatible H5 pairs using a
polychromatic projection model, beam-hardening correction, material
attenuation tables, and ODL/ASTRA fan-beam geometry.

The complete generator is not included in this initial snapshot.
`generate_pair_data.py` is intentionally a placeholder and should not be
treated as an executable data-generation workflow.

## H5 Inspection

The inspection utilities are included and can read the packaged H5 files:

```bash
python inspect_h5_data.py \
    test_data/oral_ct/test_720geo/patient_0000/000/9.h5 \
    --save-png \
    --output-dir ./h5_inspection_output
```

For HU-window visualization:

```bash
python inspect_h5_hu.py \
    test_data/oral_ct/test_720geo/patient_0000/000/9.h5 \
    --save-png \
    --output-dir ./h5_hu_output
```

Useful options include `--bit-depth 8`, `--bit-depth 16`, `--show-plot`,
`--hu-min`, `--hu-max`, and `--no-hu-window`.

## Evaluation

`calculate_psnr_ssim.py` computes PSNR and SSIM for PNG results. Set its
`gt_path` and `results_path` variables to the desired directories before
running:

```bash
python calculate_psnr_ssim.py
```

It supports one-to-one matching when result and GT counts are equal, and
many-to-one matching when multiple results share one GT.

Using the packaged `test_results_dps_rf/` and `test_data/gtdata/`, the
current script reports:

```text
Total comparisons: 10
Average PSNR: 43.1287 dB
Average SSIM: 0.9833
```

These values use the script's existing protocol: OpenCV `IMREAD_COLOR`
decoding followed by Y-channel metrics on the `[0, 255]` range. They are not
native 16-bit grayscale metrics.

## Clinical Data Format

`config/MAR_pnp_rf_clinical.yaml` documents the clinical H5 layout expected
by the loader:

```text
<clinical_data_root>/
|-- testmask.npy
|-- test_720geo_dir.txt
`-- test_720geo/
    `-- patient_XXXX/000/<image>.h5
```

Clinical H5 files should contain `ma_CT`, `ma_sinogram`, `LI_CT`,
`LI_sinogram`, and `metal_trace`. Clinical mode does not require ground
truth. The external clinical data-preparation workflow is not included.

## Reproducibility Notes

- YAML files retain paths from the original server environment and must be
  updated for another server.
- CUDA-backed projection components are initialized during pipeline startup.
- `grad_mode: full`, 512 x 512 images, and long refinement runs can require
  substantial GPU memory.
- The packaged outputs are examples associated with the included synthetic
  data, not a complete benchmark report.
- The final inference and data-generation entry points are intentionally
  withheld from this initial public snapshot.

## Methods and Software Provenance

The project uses or adapts ideas and software from:

| Component | Role | Upstream license |
| --- | --- | --- |
| JiT / SiT / LightningDiT | Rectified-flow transformer architecture and utilities | MIT |
| Diffusion Posterior Sampling | Projection-domain likelihood guidance method | Paper/method citation |
| DDIM | Sampling utilities | MIT |
| OpenAI guided-diffusion | Neural-network utilities | MIT |
| SwinIR | PSNR, SSIM, and color-conversion utilities | Apache-2.0 |
| Torch Radon | Differentiable fan-beam projection | GPL-3.0 |
| ODL and ASTRA Toolbox | CT geometry and CUDA reconstruction operators | MPL-2.0 / GPL-3.0 |
| DuDoDp-MAR | H5 data conventions and geometry provenance | No explicit upstream repository license |

Exact repositories, affected files, copyright notices, and license caveats are
recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The Diffusion Posterior Sampling and Rectified Flow method citations will be
listed together with the RefineFlow-MAR paper citation in the final release.

## Citation

The associated paper has not yet been published. Citation metadata and BibTeX
will be added after publication.

## License

Original RefineFlow-MAR contributions are released under the
[GNU General Public License v3.0](LICENSE), except where a file or component is
identified as third-party material under different terms.

Third-party components remain subject to their original licenses and copyright
notices. In particular, the upstream DuDoDp-MAR and
`hojonathanho/diffusion` repositories did not expose an explicit license when
checked on 2026-07-31; this repository does not purport to relicense their
material. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before
redistributing derived files.

Copyright (C) 2026 RefineFlow-MAR authors.
