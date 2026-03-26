# I-JEPA

A from-scratch PyTorch implementation of [I-JEPA](https://arxiv.org/abs/2301.08243) — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (Assran et al., Meta AI 2023).

## What is I-JEPA?

I-JEPA is a self-supervised learning method that learns image representations **without relying on hand-crafted data augmentations or pixel-level reconstruction**. Instead, it operates entirely in representation space:

1. A **context encoder** (ViT) processes a set of visible image patches.
2. A **target encoder** (EMA copy of the context encoder) encodes the full image, producing patch-level targets.
3. A lightweight **predictor** network takes context representations + mask tokens and predicts the target encoder's representations for the masked regions.

The model is trained to minimize the L2 distance between predictions and target representations, with no contrastive loss or image-space reconstruction required.


---

## Architecture

All components are implemented from scratch in PyTorch with no external vision library dependencies. The ViT backbone follows [Dosovitskiy et al. (2020)](https://arxiv.org/abs/2010.11929).

| Component | Description |
|---|---|
| `Tokenizer` | Splits images into non-overlapping patches using `nn.Unfold`; reconstructs with `nn.Fold` |
| `TransformerBlock` | Pre-norm transformer block with multi-head self-attention and GELU FFN |
| `Encoder` | ViT encoder with learnable positional embeddings; supports masked (subset of patches) forward pass |
| `Predictor` | Narrow transformer (384-dim) that maps context embeddings + mask tokens to target space |
| `ViT` | Full Vision Transformer for classification — wraps `Encoder` with a linear head |
| `IJEPA` | *(coming soon)* Full I-JEPA model with EMA target encoder and training logic |

---

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The `uv.lock` file is committed to the repo to ensure a reproducible environment.

```bash
git clone https://github.com/<your-username>/i-jepa.git
cd i-jepa
uv sync
```

This creates a `.venv` and installs all pinned dependencies from `uv.lock`. To activate the environment manually:

```bash
source .venv/bin/activate
```

Alternatively, prefix any command with `uv run` to use the venv without activating it:

```bash
uv run --extra dev pytest tests/ -v
```

## Running tests

```bash
uv run --extra dev pytest tests/ -v
```

---

## Usage

```python
import torch
from models import Tokenizer, Encoder, Predictor, ViT

patch_size, img_size, num_patches = 16, 224, 196

# Tokenize images into patches
tokenizer = Tokenizer(img_size=img_size, patch_size=patch_size)
images = torch.randn(4, 3, img_size, img_size)  # (B, C, H, W)
patches = tokenizer.encode(images)               # (4, 196, 768) — 196 patches

# Encode a subset of patches (context encoder)
encoder = Encoder(num_patches=num_patches, patch_size=patch_size, d_model=768, n_head=12, n_layers=12)
context_idx = torch.stack([torch.randperm(num_patches)[:100] for _ in range(4)])
ctx_embeddings = encoder(patches[..., :context_idx.shape[1], :], context_idx)  # (4, 100, 768)

# Predict target embeddings for masked positions
predictor = Predictor(num_patches=num_patches, encoder_dim=768, d_model=384, n_head=6, n_layers=6)
target_idx = torch.stack([torch.randperm(num_patches)[:20] for _ in range(4)])
predictions = predictor(ctx_embeddings, context_idx, target_idx)  # (4, 20, 768)

# ViT for classification (takes pre-tokenized patches)
vit = ViT(num_patches=num_patches, num_classes=1000, patch_size=patch_size, n_head=12, n_layers=12)
logits = vit(patches)  # (4, 1000)
```


---

## Design Decisions

### Positional Embeddings

We use **learnable positional embeddings** (added to patch tokens after projection), matching the original I-JEPA and ViT papers.

We considered two alternatives:

**ALiBi** — adds a distance-based negative bias to attention scores, requiring no learnable parameters and generalising to unseen sequence lengths. Rejected because its distance penalty suppresses attention between far-apart patches. In I-JEPA, context and target blocks are deliberately non-overlapping and spatially distant; suppressing that cross-region attention directly undermines the predictor's task.

**RoPE (Rotary Position Embeddings)** — encodes position by rotating Q/K vectors, providing relative position information without a distance penalty. Distant patches can still attend freely, and the position is baked into each query/key so mask tokens at target locations are inherently position-aware. RoPE is the strongest candidate for a future improvement, particularly if we want resolution generalisation (training at 224×224, inferring at higher resolutions).

For now, learnable embeddings keep the implementation faithful to the paper. RoPE can be explored as a future enhancement.

---

## Paper

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}

@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
