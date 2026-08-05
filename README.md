<div align="center">

# RefineFlow-MAR: Physics-Guided Unsupervised CBCT Metal Artifact Reduction

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white"/>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-ee4c2c?logo=pytorch&logoColor=white"/>
  <img alt="CUDA" src="https://img.shields.io/badge/Compute-NVIDIA%20CUDA-76b900?logo=nvidia&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/badge/License-GPL--3.0-2f855a"/>
  <img alt="Release" src="https://img.shields.io/badge/Release-Initial%20Snapshot-555555"/>
</p>

<p>
  <b>Paper:</b> Manuscript &nbsp;|&nbsp;
  <b>Task:</b> Dental CBCT Metal Artifact Reduction
</p>

RefineFlow-MAR learns an anatomical rectified-flow prior from unpaired clean
CBCT and couples it with continuous projection-domain physical guidance.

</div>

---

## Visual Results

The moving divider reveals the RefineFlow-MAR reconstruction on the left while
retaining the metal-affected input on the right.

<table>
  <tr>
    <th align="center">Case 1</th>
    <th align="center">Case 2</th>
    <th align="center">Case 3</th>
  </tr>
  <tr>
    <td align="center"><img src="assets/demos/case_01.gif" width="260" alt="RefineFlow-MAR result for dental CBCT case 1"/></td>
    <td align="center"><img src="assets/demos/case_02.gif" width="260" alt="RefineFlow-MAR result for dental CBCT case 2"/></td>
    <td align="center"><img src="assets/demos/case_03.gif" width="260" alt="RefineFlow-MAR result for dental CBCT case 3"/></td>
  </tr>
</table>

## Highlights

- **Unpaired anatomical prior:** learns the clean CBCT distribution without
  requiring paired metal-corrupted and artifact-free scans.
- **Continuous physics guidance:** injects reliability-masked Radon-domain
  gradients throughout the rectified-flow trajectory instead of applying a
  single hard projection correction.
- **Risk-aware refinement:** combines gradient correction near unreliable
  regions with sinogram replacement in reliable anatomy.
- **Dual-domain consistency:** balances artifact suppression, anatomical
  plausibility, and fidelity to the measured projections.

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
- `train_rectified_flow.py` is a public placeholder; the final training entry
  point will be released after publication of the associated paper.
- The pretrained JiT checkpoint is distributed separately and is not currently
  available from this repository.
- The paper citation will be added after publication.

As a result, this snapshot documents the intended workflows but does not yet
provide complete end-to-end training or inference commands.

## Repository Layout

```text
RefineFlow-MAR/
|-- assets/demos/                         # Animated visual comparisons
|-- config/
|   |-- MAR_pnp_rf.yaml
|   |-- MAR_pnp_rf_clinical.yaml
|   `-- train_JiT_B16_1ch_rf_fp32_cbct.yaml
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
|-- train_rectified_flow.py               # Public-release placeholder
|-- inspect_h5_data.py
|-- inspect_h5_hu.py
|-- calculate_psnr_ssim.py
|-- requirements.txt
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
extension.

Most package versions are intentionally not pinned because CUDA servers
require compatible PyTorch, torchvision, compiler, and driver combinations.
For strict reproduction, record the working server environment with
`python -m pip freeze` and the CUDA/PyTorch version information.

## Training

The repository includes the single-channel JiT rectified-flow model components
and the intended training configuration in
[config/train_JiT_B16_1ch_rf_fp32_cbct.yaml](config/train_JiT_B16_1ch_rf_fp32_cbct.yaml).
The configuration records the model architecture, preprocessing mode,
optimization settings, EMA parameters, and checkpoint directory used by the
project.

The complete training entry point is not included in this initial snapshot.
`train_rectified_flow.py` is intentionally a placeholder and should not be
treated as an executable training workflow. It will be replaced after
publication of the associated paper.

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
contain patient or author attributes. Users are responsible for any replacement
data they provide.

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
- The final training, inference, and data-generation entry points are
  intentionally withheld from this initial public snapshot.

## Citation

The associated paper has not yet been published. Citation metadata and BibTeX
will be added after publication.

## License

RefineFlow-MAR is released under the
[GNU General Public License v3.0](LICENSE).

Copyright (C) 2026 RefineFlow-MAR authors.
