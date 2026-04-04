import math
import logging
from collections import namedtuple
from multiprocessing import Value
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset, load_from_disk
from PIL import Image


logger = logging.getLogger(__name__)
_DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
):
    """
    Build the pre-training image transform for I-JEPA.

    By design the paper uses *minimal* augmentation — just a random resized
    crop and normalisation — to show that the learned representations do not
    depend on hand-crafted data augmentations.
    """
    logger.info("Building data transforms")

    normalization = (0, 1) if normalization is None else normalization

    return transforms.Compose(([] if crop_size is None else [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]) + [
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1]),
    ])

class MaskCollator:
    """
    Custom collate function that creates encoder (context) and predictor
    (target) masks.

    Returns:
        collated_images : Tensor  [B, C, H, W]
        enc_masks       : list of Tensor  (nenc tensors, each [B, n_keep_enc])
        pred_masks      : list of Tensor  (npred tensors, each [B, n_keep_pred])

    Mask tensors contain *patch indices* (into the flattened H×W grid) that
    should be kept / predicted.
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=4,
        allow_overlap=False,
    ):
        if not isinstance(input_size, tuple): input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value("i", -1)

    def step(self):
        """Atomically increment the iteration counter and return the value."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        """Sample a block (h, w) in *patch* units from scale and AR ranges."""
        _rand = torch.rand(1, generator=generator).item()

        # block scale → number of patches
        min_s, max_s = scale
        max_keep = int(self.height * self.width * (min_s + _rand * (max_s - min_s)))

        # block aspect ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # clamp to grid size
        return (
            min(int(round(math.sqrt(max_keep * aspect_ratio))), self.height), 
            min(int(round(math.sqrt(max_keep / aspect_ratio))), self.width)
        )

    def _sample_block_mask(self, block_size, acceptable_regions=[], tries=20):
        """
        Sample a rectangular block at a random position on the patch grid.

        Args:
            block_size: (h, w) block size in patch units.
            acceptable_regions: optional list of 2‑D binary masks that the
                sampled block must partially lie within (used to remove
                overlap with predictor blocks from the encoder mask).

        Returns:
            mask            : 1-D tensor of flattened patch indices inside the block
            mask_complement : 2-D binary tensor (H, W) with 0 inside the block, 1 outside
        """
        h, w = block_size
        valid_patch_mask = math.prod(acceptable_regions)

        for _ in range(tries):
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
            
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top: top + h, left: left + w] = 1
            mask *= valid_patch_mask
            
            mask_indices = torch.nonzero(mask.flatten()).flatten()
            
            if mask_indices.numel() >= self.min_keep: break
        else:
            if acceptable_regions:
                logger.warning(f"MaskCollator: valid mask not found, relaxing acceptable regions [{len(acceptable_regions) - 1}]") 
                
                return self._sample_block_mask(block_size, acceptable_regions[:-1], tries)

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top: top + h, left: left + w] = 0

        return mask_indices, mask_complement

    def __call__(self, batch):
        """
        Collate a list of (image, label) tuples into a batch *and* generate
        encoder / predictor masks.

        Steps:
            1. Sample predictor block size using a per-batch seed.
            2. Sample encoder block size using the same seed (aspect ratio = 1).
            3. For each image, sample `npred` predictor block positions.
            4. For each image, sample `nenc` encoder block positions,
               constrained to *not* overlap with the predictor blocks.
            5. Truncate masks to the minimum number of kept patches across
               the batch so they can be stacked.

        Returns:
            collated_batch  : Tensor [B, C, H, W]
            collated_enc    : list[Tensor]  length nenc, each [B, min_keep_enc]
            collated_pred   : list[Tensor]  length npred, each [B, min_keep_pred]
        """
        # -- shared per-batch seed for block *sizes*
        g = torch.Generator()
        g.manual_seed(self.step())
        
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1.0, 1.0),
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.height * self.width

        for _ in range(len(batch)):
            # predictor (target) masks
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            # encoder (context) masks — no overlap with predictor blocks
            acceptable_regions = [] if self.allow_overlap else masks_C
            masks_e = []
            for _ in range(self.nenc):
                mask = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)[0]
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # truncate to min kept across the batch so tensors are stackable
        return (
            torch.utils.data.default_collate(batch), 
            torch.stack(torch.utils.data.default_collate([
                [cm[:min_keep_enc] for cm in cm_list] 
                for cm_list in collated_masks_enc
            ])), 
            torch.stack(torch.utils.data.default_collate([
                [cm[:min_keep_pred] for cm in cm_list] 
                for cm_list in collated_masks_pred
            ]))
        )

def apply_masks(x, masks):
    """
    Select patch embeddings at the positions given by *masks*.

    Args:
        x     : Tensor of shape [B, N, D]  (patch embeddings)
        masks : list of Tensor, each [B, n_keep]  (patch indices to keep)

    Returns:
        Tensor of shape [len(masks)*B, n_keep, D]
    """
    return torch.cat(
        [torch.gather(x, dim=1, index=m.unsqueeze(-1).repeat(1, 1, x.size(-1))) for m in masks],
        dim=0
    )

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform, patcher=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.patcher = patcher

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        img = self.transform(example["image"].convert("RGB"))
        return (
            self.patcher(img.unsqueeze(0)).squeeze(0) if self.patcher else img,
            example["label"]
        )

def make_imagenet(
    transform,
    patcher=None,
    batch_size=1,
    dataset_name=None,
    local_name=None,
    collator=None,
    pin_mem=True,
    shuffle=True,
    num_workers=os.cpu_count(),
    split="train",
    drop_last=True,
):
    """
    Build a mini-ImageNet dataset and DataLoader from Hugging Face.

    Downloads the dataset on first use and caches it locally under
    ``datasets/<local_name>/<split>/``.

    Args:
        transform     : torchvision transform to apply to each image.
        patcher       : module to patch the image.
        batch_size    : per-GPU batch size.
        dataset_name  : Hugging Face dataset identifier.
        local_name    : folder name inside ``datasets/`` for caching.
        collator      : custom collate_fn (e.g. MaskCollator).
        pin_mem       : pin memory for faster GPU transfer.
        shuffle       : shuffle the dataset.
        num_workers   : number of DataLoader workers.
        split         : one of ``train``, ``validation``, or ``test``.
        drop_last     : drop the last incomplete batch.

    Returns:
        data_loader : torch.utils.data.DataLoader
    """
    if local_name is None:
        if dataset_name is None: raise ValueError("Provide at least one of local_name or dataset_name.")
        local_name = dataset_name.split("/")[-1]

    split_dir = os.path.join(_DATASETS_DIR, local_name, split)

    if os.path.isdir(split_dir):
        hf_dataset = load_from_disk(split_dir)
        
        logger.info(f"Dataset loaded from disk — {len(hf_dataset)} images ({local_name}, {split})")
    else:
        if dataset_name is None: raise FileNotFoundError(f"No cached dataset at {split_dir}. Provide dataset_name to download it first.")
        
        hf_dataset = load_dataset(dataset_name, split=split)
        os.makedirs(split_dir, exist_ok=True)
        hf_dataset.save_to_disk(split_dir)
        
        logger.info(f"Dataset downloaded and saved — {len(hf_dataset)} images ({dataset_name}, {split})")

    data_loader = torch.utils.data.DataLoader(
        ImageNetDataset(
            hf_dataset,
            transform,
            patcher
        ),
        collate_fn=collator,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    
    logger.info("DataLoader created")

    return data_loader

Label = namedtuple("Label", ["name", "id", "color"])

labels = [
    Label("road", 0, (128, 64, 128)),
    Label("sidewalk", 1, (244, 35, 232)),
    Label("building", 2, (70, 70, 70)),
    Label("wall", 3, (102, 102, 156)),
    Label("fence", 4, (190, 153, 153)),
    Label("pole", 5, (153, 153, 153)),
    Label("traffic light", 6, (250, 170, 30)),
    Label("traffic sign", 7, (220, 220, 0)),
    Label("vegetation", 8, (107, 142, 35)),
    Label("terrain", 9, (152, 251, 152)),
    Label("sky", 10, (70, 130, 180)),
    Label("person", 11, (220, 20, 60)),
    Label("rider", 12, (255, 0, 0)),
    Label("car", 13, (0, 0, 142)),
    Label("truck", 14, (0, 0, 70)),
    Label("bus", 15, (0, 60, 100)),
    Label("train", 16, (0, 80, 100)),
    Label("motorcycle", 17, (0, 0, 230)),
    Label("bicycle", 18, (119, 11, 32)),
    Label("unlabeled", 255, (0, 0, 0)),
]

NUM_CLASSES = len(labels) - 1
TRAINID_TO_COLOR = {lb.id: lb.color for lb in labels}
TRAINID_TO_NAME = {lb.id: lb.name for lb in labels}
BDD_PIL_SIZE: Tuple[int, int] = (1280, 720)


def _bdd_image_paths(images_dir: Union[str, Path]) -> List[Path]:
    """
    Sorted ``*.jpg`` under ``images_dir``; keep only files whose PIL size is
    ``BDD_PIL_SIZE``.
    """
    all_jpg = sorted(Path(images_dir).glob("*.jpg"))
    kept = []
    for img_path in all_jpg:
        with Image.open(img_path) as img:
            if img.size == BDD_PIL_SIZE: kept.append(img_path)
    logger.info(
        "BDD image list: kept %d / %d under %s%s",
        len(kept),
        len(all_jpg),
    )
    return kept


class BDDDataset(torch.utils.data.Dataset):
    """
    BDD images at ``BDD_PIL_SIZE`` (see ``_bdd_image_paths``).

    Each sample is built as ``np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0``, then a
    ``torch`` tensor, then ``transform`` (use ``make_transforms(..., to_tensor=False)`` so you do not
    run ``ToTensor`` twice).

    * ``labels_dir is None``: pre-training — returns ``(tensor, 0)`` for ``MaskCollator``.
    * ``labels_dir`` set: segmentation — returns ``(image_tensor, mask_long)``; mask is never transformed.
    """

    def __init__(
        self,
        image_paths: List[Path],
        transform: Callable[..., torch.Tensor],
        patcher=None,
        labels_dir: Optional[Union[str, Path]] = None,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.patcher = patcher
        self.labels_dir = Path(labels_dir) if labels_dir else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as im:
            img = self.transform(torch.from_numpy(np.ascontiguousarray(np.array(im.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0)))
        out = self.patcher(img.unsqueeze(0)).squeeze(0) if self.patcher else img
        mask = torch.from_numpy(np.array(Image.open(self.labels_dir / f"{img_path.stem}_train_id.png"), dtype=np.uint8)).long() if self.labels_dir else None
        return out, mask


def make_bdd_dataloader(
    mode: str,
    split: str,
    transform: Callable[..., torch.Tensor],
    batch_size: int = 1,
    patcher=None,
    collator=None,
    pin_mem: bool = True,
    shuffle: bool = True,
    num_workers: int = os.cpu_count(),
    drop_last: bool = True,
):
    """
    DataLoader for local BDD layout: ``<bdd_root>/{100k|10k}/images/{split}/``.

    Args:
        mode: ``"pretrain"`` → ``100k`` images only; ``"segmentation"`` → ``10k``
            images + ``labels/{split}/*_train_id.png`` (``split`` must be ``train`` or ``val``).
        split: ``train``, ``val``, or ``test`` (``test`` only for pretrain).
        transform: Required. ``BDDDataset`` converts PIL to ``float32`` CHW ``[0,1]`` then calls this;
            use ``make_transforms(crop_size=None, to_tensor=False)`` so ``Normalize`` runs without ``ToTensor``.
        patcher: Used only for ``pretrain`` (e.g. tokenizer ``encode``).
        collate_fn: e.g. ``MaskCollator`` for pretrain.
        bdd_root: Defaults to ``datasets/BDD`` next to this package.
    """
    mode = mode.lower()
    if mode == "pretrain":
        subset = "100k"
        if split not in ("train", "val", "test"): raise ValueError(f"pretrain split must be train, val, or test; got {split!r}")
    elif mode == "segmentation":
        subset = "10k"
        if split not in ("train", "val"): raise ValueError(f"segmentation split must be train or val; got {split!r}")
    else: raise ValueError(f"mode must be 'pretrain' or 'segmentation'; got {mode!r}")

    bdd_root = os.path.join(_DATASETS_DIR, "BDD")

    data_loader = torch.utils.data.DataLoader(
        dataset=BDDDataset(
            image_paths=_bdd_image_paths(os.path.join(bdd_root, subset, "images", split)),
            transform=transform,
            patcher=patcher,
            labels_dir=None if mode == "pretrain" else os.path.join(bdd_root, subset, "labels", split)
        ),
        collate_fn=collator,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers,
    )

    logger.info("DataLoader created")
    
    return data_loader


def _overlay_mask_on_image(image, mask_indices, patch_size, grid_size, darken=0.65):
    """Overlay a mask on an image, only for visualization."""
    grid_h, grid_w = grid_size
    mask_indices = mask_indices.flatten().to(torch.long)
    mask_grid = torch.zeros((grid_h, grid_w), device=image.device, dtype=torch.float32)
    mask_grid[torch.div(mask_indices, grid_w, rounding_mode="floor"), mask_indices % grid_w] = 1.0
    mask_pixels = mask_grid.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)[: image.shape[1], : image.shape[2]]
    return image * mask_pixels * darken

def main():
    from models import Tokenizer
    import matplotlib.pyplot as plt

    tokenizer = Tokenizer(img_size=224, patch_size=16)
    collator = MaskCollator()

    data_loader = make_imagenet(
        dataset_name="timm/mini-imagenet", # only specify the first time to download
        local_name="mini-imagenet",
        transform=make_transforms(normalization=(0, 1)), # for visuslization, no normalization
        patcher=tokenizer.encode,
        collator=collator,
        split="test"
    )

    for (images, labels), enc_masks, pred_masks in data_loader:
        print(apply_masks(images, enc_masks).shape)
        print(apply_masks(images, pred_masks).shape)

        masks_to_plot = []
        for i, mask in enumerate(enc_masks): masks_to_plot.append((f"enc {i}", mask[0]))
        for i, mask in enumerate(pred_masks): masks_to_plot.append((f"pred {i}", mask[0]))

        total = len(masks_to_plot)
        cols = min(total, 5)
        rows = math.ceil(total / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        for ax, (title, mask_indices) in zip(axes, masks_to_plot):
            overlay = _overlay_mask_on_image(
                image=tokenizer.decode(images)[0],
                mask_indices=mask_indices,
                patch_size=collator.patch_size,
                grid_size=(collator.height, collator.width),
            )
            ax.imshow(overlay.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_title(title)
            ax.axis("off")

        for ax in axes[total:]: ax.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__": #main()
    data_loader = make_bdd_dataloader(
        mode="pretrain",
        split="train",
        transform=make_transforms(normalization=(0, 1)),
        patcher=None,
        collator=None,
        pin_mem=True,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True,
    )
    for images, masks in data_loader:
        print(images.shape)
        print(masks.shape)
        break