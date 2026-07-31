# Third-Party Notices

RefineFlow-MAR incorporates, adapts, or depends on work from the projects
listed below. The project-level GPL-3.0 license does not replace the original
copyright notices or license terms that apply to third-party material.

This notice is a provenance record, not legal advice. Where an upstream
repository does not publish a license, this file does not grant permission on
behalf of that upstream author.

## Incorporated or adapted source

### SwinIR

- Project: https://github.com/JingyunLiang/SwinIR
- License: Apache License 2.0
- Relevant file: `utils/metrics.py`
- Use: PSNR, SSIM, color-conversion, and image metric utilities.
- Local copy of license: `licenses/Apache-2.0.txt`

Copyright 2021 SwinIR Authors.

### DDIM

- Project: https://github.com/ermongroup/ddim
- License: MIT
- Relevant file: `utils/sampling.py`
- Use: DDIM sampling utilities adapted for this project.

Copyright (c) 2020 Jiaming Song.

### LightningDiT

- Project: https://github.com/hustvl/LightningDiT
- License: MIT
- Relevant file: `patch_diffusion/jit/util/model_util.py`
- Use: rotary embeddings, RMS normalization, and positional-embedding helpers.

Copyright (c) 2024 HUST Vision Lab.

### SiT

- Project: https://github.com/willisma/SiT
- License: MIT
- Relevant file: `patch_diffusion/jit/model_jit_rf.py`
- Use: diffusion/flow transformer design referenced by the JiT implementation.

Copyright (c) Meta Platforms, Inc. and affiliates.

### JiT

- Closest public source match: https://github.com/Yassin1-prog/JIT
- License in that repository: MIT
- Relevant file: `patch_diffusion/jit/model_jit_rf.py`
- Use: JiT architecture adapted from RGB images to single-channel CT.

Copyright (c) 2025 Tianhong Li.

### OpenAI guided-diffusion

- Project: https://github.com/openai/guided-diffusion
- License: MIT
- Relevant file: `patch_diffusion/nn.py`
- Use: common diffusion neural-network utilities.

Copyright (c) 2021 OpenAI.

### MIT License Text

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The applicable copyright notice above and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Upstream sources without an explicit repository license

### DuDoDp-MAR

- Project: https://github.com/DeepXuan/DuDoDp-MAR
- License status checked on 2026-07-31: no license file or GitHub license
  declaration was present.
- Relevant areas: `geometry/`, the DuDoDp-compatible H5 layout, and portions
  of the physical MAR data workflow.

The RefineFlow-MAR GPL-3.0 license applies only to modifications and original
contributions for which the RefineFlow-MAR authors hold rights. It does not
purport to relicense upstream DuDoDp-MAR material. Redistribution permission
for that upstream material should be confirmed with its authors.

### hojonathanho/diffusion

- Project: https://github.com/hojonathanho/diffusion
- License status checked on 2026-07-31: no license file or GitHub license
  declaration was present.
- Relevant file: `patch_diffusion/nn.py`
- Use: the modified timestep-embedding formula cites this implementation.

This citation identifies provenance and does not assign a license to the
upstream repository.

## Runtime dependencies

The repository does not vendor these dependency sources, but the pipeline
uses them at runtime:

- Torch Radon: https://github.com/matteo-ronchetti/torch-radon, GPL-3.0.
- ODL: https://github.com/odlgroup/odl, Mozilla Public License 2.0.
- ASTRA Toolbox: https://github.com/astra-toolbox/astra-toolbox, GPL-3.0.
- PyTorch and torchvision: their respective upstream licenses.

See each dependency distribution for its complete and current license terms.

