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
| `PatchEmbedding` | Linear projection of flattened patches into embedding dimension |
| `TransformerEncoderBlock` | Pre-norm transformer block with multi-head self-attention and GELU FFN |
| `ViT` | Full Vision Transformer encoder — backbone for context and target encoders |
| `Predictor` | *(coming soon)* Narrow transformer that maps context embeddings + mask tokens to target space |
| `IJEPA` | *(coming soon)* Full I-JEPA model with EMA target encoder and training logic |

---

## Installation

```bash
git clone https://github.com/<your-username>/i-jepa.git
cd i-jepa
pip install torch
```

No additional dependencies required.

---

## Usage

```python
import torch
from vit import ViT

model = ViT(
    img_size=224,
    patch_size=16,
    n_embed=768,
    n_head=12,
    n_layers=12,
)

x = torch.randn(4, 3, 224, 224)  # (batch, channels, H, W)
embeddings = model(x)             # (4, 196, 768) — 196 patches of dim 768
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
