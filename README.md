# VisualSplit

> **TL;DR** — Learn an interpretable image representation by **splitting** an image into **edge**, **color-segmentation**, and **gray-level histogram** descriptors, then **reconstruct** it from only those descriptors. Useful for reconstruction/restoration and controllable editing.

<p align="left">
  <a href="https://pypi.org/project/torch/"><img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c" alt="PyTorch"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <a href="https://huggingface.co/quchenyuan/VisualSplit"><img src="https://img.shields.io/badge/Weights-HuggingFace-yellow.svg" alt="HuggingFace Weights"></a>
</p>

---

## Table of Contents
- [Highlights](#highlights)
- [What's inside](#whats-inside)
- [Install](#install)
- [Quickstart](#quickstart)
  - [Load pretrained & reconstruct](#load-pretrained--reconstruct)
  - [Train from scratch](#train-from-scratch)
  - [Validate restoration](#validate-restoration)
- [Config & Data](#config--data)
- [Roadmap](#roadmap)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Highlights
- **Interpretable**: decouples geometry, color regions, and global tone into separate, human-understandable descriptors.
- **Mask-free pretraining**: descriptors themselves act as “information-sparse” inputs; no patch masking tricks.
- **Pretrained checkpoint**: ready-to-use weights on HF → **[quchenyuan/VisualSplit](https://huggingface.co/quchenyuan/VisualSplit)**.
- **Restoration validation**: basic examples to evaluate reconstruction/PSNR/SSIM on your data.

---

## What's inside

```
VisualSplit/
├─ visualsplit/
│  ├─ models/
│  │  └─ CrossViT.py                 # ViT-based multi-modal encoder + lightweight decoder
│  ├─ pipeline/
│  │  └─ train_CrossViT.py           # self-supervised training (reconstruction objective)
│  ├─ utils/
│  │  └─ feature_extractor.py        # edge / color segmentation / gray histogram
│  └─ ...
├─ LICENSE
└─ README.md
```

- **Descriptors**: Sobel edges, K-means color segmentation (LAB), 100-bin gray-level histogram.
- **Encoder**: ViT backbone consumes **edge+seg** as patch tokens; **histogram** enters via global conditioning.
- **Decoder**: lightweight head to reconstruct RGB.

---

## Install

```bash
# clone
git clone https://github.com/HenryQUQ/VisualSplit.git
cd VisualSplit

# (option A) pip editable install
pip install -e .

# (option B) poetry
# poetry install
```

> Requires Python ≥ 3.10, PyTorch ≥ 2.2 (CUDA recommended). See `requirements.txt` / `pyproject.toml` for full deps.

---

## Quickstart

### Load pretrained & reconstruct

```python
import torch
from PIL import Image
from torchvision import transforms

from visualsplit.models.CrossViT import CrossViTForPreTraining, CrossViTConfig
# If your project structure differs, adjust this import path accordingly:
from visualsplit.utils.feature_extractor import FeatureExtractor

# 1) create model (match training config)
config = CrossViTConfig(image_size=224, patch_size=16)
model = CrossViTForPreTraining(config).eval()

# 2) load weights (download from HF manually or via huggingface_hub)
# from huggingface_hub import hf_hub_download
# ckpt_path = hf_hub_download(repo_id="quchenyuan/VisualSplit", filename="visualsplit_vitb.safetensors")
state = torch.load("path/to/VisualSplit_checkpoint.pth", map_location="cpu")
model.load_state_dict(state)

# 3) prepare image & descriptors
image = Image.open("my_test_image.jpg").convert("RGB")
to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
img = to_tensor(image).unsqueeze(0)
extractor = FeatureExtractor()  # returns edge, hist, segmented, (optional ab)
edge, gray_hist, segmented, _ = extractor(img)

# 4) reconstruct
with torch.no_grad():
    outputs = model(
        source_edge=edge,
        source_gray_level_histogram=gray_hist,
        source_segmented_rgb=segmented
    )
recon = outputs["logits_reshape"].clamp(0, 1)  # (1,3,224,224)

# 5) save
transforms.ToPILImage()(recon.squeeze(0)).save("reconstructed.png")
```

### Train from scratch

> Run from repo root to ensure imports work.

```bash
# single GPU
python -m visualsplit.pipeline.train_CrossViT   --dataset ImageNet-1k-pure   --batch_size 64 --epochs 100 --learning_rate 1.5e-4

# or with accelerate (if configured)
# accelerate launch -m visualsplit.pipeline.train_CrossViT --dataset ImageNet-1k-pure ...
```

The script:
- loads data (HF datasets or your custom loader),
- extracts descriptors on-the-fly (with caching),
- optimizes reconstruction (MSE + LPIPS),
- saves logs/checkpoints (default under `cache/logs/`).

### Validate restoration

Use the pretrained model to **reconstruct** from descriptors extracted on **your degraded images** (e.g., noisy or low-light). Compare outputs vs. ground-truth with PSNR/SSIM using your evaluation pipeline of choice. The same reconstruction snippet above can be adapted into a loop over a dataset to compute metrics.

---

## Config & Data

- **Backbone**: ViT-B by default (`image_size=224`, `patch_size=16`).
- **Descriptors**: LAB→(Sobel on L, 100-bin hist on L, K-means on AB).  
- **Dataset**: default uses ImageNet-1K via HF; you can plug in any image folder dataset as long as it yields tensors to `FeatureExtractor`.
- **Hardware**: training prefers ≥16GB GPU; inference works on CPU but is faster on GPU.

---

## Roadmap

- [ ] **Google Colab**: interactive demo (extract descriptors → reconstruct).
- [ ] **HuggingFace Space**: web UI to upload, view descriptors, and reconstruct.
- [x] **Pretrained checkpoint** on HF: https://huggingface.co/quchenyuan/VisualSplit
- [x] **Training script** & **restoration validation** basics.

---

## FAQ / Troubleshooting

**Q: `ImportError: attempted relative import with no known parent package`?**  
A: Run from repo root and use module mode:  
`python -m visualsplit.pipeline.train_CrossViT ...`

**Q: Where do checkpoints go / how to change?**  
A: Check the training script args (save dir/log dir flags) and set your preferred path.

**Q: My reconstruction looks too dark/bright.**  
A: Ensure inputs are resized to the training size (default 224) and histogram extraction matches training (100 bins on L channel).

---

## License
Apache-2.0. See [LICENSE](./LICENSE).

---

## Acknowledgements
Built with PyTorch & the HF ecosystem; classic CV ops (Sobel/K-Means) via common libs. Thanks to collaborators and the community.
