import math
import logging
from collections import namedtuple
from multiprocessing import Value
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.io as tvio
from datasets import load_dataset, load_from_disk


logger = logging.getLogger(__name__)
_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

def make_transforms(
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    crop_scale: Tuple[float, float] = (0.3, 1.0),
    normalization: Tuple = (0, 1),
) -> transforms.Compose:
    """Build the pre-training image transform for I-JEPA.

    By design the paper uses *minimal* augmentation — just a random resized
    crop and normalisation — to show that the learned representations do not
    depend on hand-crafted data augmentations.

    Args:
        crop_size: Output spatial size for ``RandomResizedCrop``.  If ``None``
            the crop step is skipped.
        crop_scale: Range of area ratios for the random crop.
        normalization: ``(mean, std)`` pair passed to ``transforms.Normalize``.

    Returns:
        A composed torchvision transform.
    """
    logger.info("Building data transforms")

    return transforms.Compose(
        ([transforms.RandomResizedCrop(crop_size, scale=crop_scale)] if crop_size else [])
        + [transforms.ConvertImageDtype(torch.float32), transforms.Normalize(*normalization)]
    )

class MaskCollator:
    """Custom collate function that creates encoder (context) and predictor (target) masks.

    Calling an instance returns ``(collated_images, enc_masks, pred_masks)``:

    * ``collated_images``: Tensor ``[B, C, H, W]``
    * ``enc_masks``: list of ``nenc`` tensors, each ``[B, n_keep_enc]``
    * ``pred_masks``: list of ``npred`` tensors, each ``[B, n_keep_pred]``

    Mask tensors contain *patch indices* (into the flattened H x W grid) that
    should be kept / predicted.
    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]],
        patch_size: int = 16,
        enc_mask_scale: Tuple[float, float] = (0.85, 1.0),
        pred_mask_scale: Tuple[float, float] = (0.15, 0.2),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
        nenc: int = 1,
        npred: int = 4,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ) -> None:
        """Initialise mask geometry from image and patch sizes.

        Args:
            input_size: Spatial resolution of the input image (pixels).
            patch_size: Side length of each square patch (pixels).
            enc_mask_scale: Area-ratio range for the encoder (context) block.
            pred_mask_scale: Area-ratio range for the predictor (target) block.
            aspect_ratio: Aspect-ratio range for sampled blocks.
            nenc: Number of encoder masks per image.
            npred: Number of predictor masks per image.
            min_keep: Minimum number of patches a valid mask must contain.
            allow_overlap: If ``True``, encoder and predictor masks may overlap.
        """
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

    def step(self) -> int:
        """Atomically increment the iteration counter and return the value."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        
        return v

    def _sample_block_size(
        self,
        generator: torch.Generator,
        scale: Tuple[float, float],
        aspect_ratio_scale: Tuple[float, float],
    ) -> Tuple[int, int]:
        """Sample a block ``(h, w)`` in *patch* units from scale and AR ranges.

        Args:
            generator: PyTorch random generator for reproducibility.
            scale: ``(min, max)`` area-ratio range relative to the full grid.
            aspect_ratio_scale: ``(min, max)`` aspect-ratio range.

        Returns:
            ``(h, w)`` block size in patch units, clamped to the grid.
        """
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

    def _sample_block_mask(
        self,
        block_size: Tuple[int, int],
        acceptable_regions: List[torch.Tensor] = [],
        tries: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a rectangular block at a random position on the patch grid.

        Args:
            block_size: ``(h, w)`` block size in patch units.
            acceptable_regions: Optional list of 2-D binary masks that the
                sampled block must partially lie within (used to remove
                overlap with predictor blocks from the encoder mask).
            tries: Maximum number of random placement attempts.

        Returns:
            A ``(mask, mask_complement)`` tuple where *mask* is a 1-D tensor of
            flattened patch indices inside the block and *mask_complement* is a
            2-D binary tensor ``(H, W)`` with 0 inside the block and 1 outside.
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

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate ``(image, label)`` tuples and generate encoder / predictor masks.

        Steps:
            1. Sample predictor block size using a per-batch seed.
            2. Sample encoder block size using the same seed (aspect ratio = 1).
            3. For each image, sample ``npred`` predictor block positions.
            4. For each image, sample ``nenc`` encoder block positions,
               constrained to *not* overlap with the predictor blocks.
            5. Truncate masks to the minimum number of kept patches across
               the batch so they can be stacked.

        Args:
            batch: List of ``(image, label)`` tuples from the dataset.

        Returns:
            A ``(collated_batch, collated_enc, collated_pred)`` tuple where
            *collated_batch* is ``[B, C, H, W]``, *collated_enc* is a list of
            ``nenc`` tensors each ``[B, min_keep_enc]``, and *collated_pred* is
            a list of ``npred`` tensors each ``[B, min_keep_pred]``.
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

def apply_masks(x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Select patch embeddings at the positions given by *masks*.

    Args:
        x: Patch embeddings of shape ``[B, N, D]``.
        masks: A tensor ``[M, B, n_keep]`` containing patch indices.

    Returns:
        Tensor of shape ``[M, B, n_keep, D]``.
    """
    return torch.take_along_dim(x[None, ...], masks[..., None], dim=2)

IMAGENET_SIZE: Tuple[int, int] = (224, 224)

IMAGENET_NORMALIZATION = (
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
)

class ImageNetDataset(torch.utils.data.Dataset):
    """Thin wrapper around a Hugging Face image-classification dataset.

    Each example is expected to have ``"image"`` (PIL) and ``"label"`` (int)
    keys, matching the layout of datasets like ``timm/mini-imagenet``.
    """

    def __init__(
        self,
        hf_dataset: Any,
        transform: Callable[..., torch.Tensor],
        patcher: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.patcher = patcher

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        example = self.hf_dataset[idx]
        img = self.transform(transforms.functional.pil_to_tensor(example["image"].convert("RGB")))
        return (
            self.patcher(img.unsqueeze(0)).squeeze(0) if self.patcher else img,
            example["label"]
        )

def make_imagenet(
    split: str,
    transform: Callable[..., torch.Tensor],
    patcher: Optional[Callable[..., torch.Tensor]] = None,
    collator: Optional[Callable] = None,
    batch_size: int = 1,
    pin_mem: bool = True,
    shuffle: bool = True,
    num_workers: int = 1,
    drop_last: bool = True,
    prefetch_factor: int = 1,
    dataset_name: Optional[str] = None,
    local_name: Optional[str] = None,
) -> torch.utils.data.DataLoader:
    """Build a mini-ImageNet dataset and DataLoader from Hugging Face.

    Downloads the dataset on first use and caches it locally under
    ``datasets/<local_name>/<split>/``.

    Args:
        split: One of ``"train"``, ``"validation"``, or ``"test"``.
        transform: Torchvision transform to apply to each image.
        patcher: Optional module to patchify the image.
        collator: Custom ``collate_fn`` (e.g. ``MaskCollator``).
        batch_size: Per-GPU batch size.
        pin_mem: Pin memory for faster GPU transfer.
        shuffle: Shuffle the dataset.
        num_workers: Number of DataLoader workers.
        drop_last: Drop the last incomplete batch.
        prefetch_factor: Number of batches to prefetch.
        dataset_name: Hugging Face dataset identifier.
        local_name: Folder name inside ``datasets/`` for caching.

    Returns:
        Configured ``torch.utils.data.DataLoader``.
    """
    if local_name is None:
        if dataset_name is None: raise ValueError("Provide at least one of local_name or dataset_name.")
        local_name = dataset_name.split("/")[-1]

    split_dir = _DATASETS_DIR / local_name / split

    if split_dir.is_dir(): hf_dataset = load_from_disk(split_dir)
    else:
        if dataset_name is None: raise FileNotFoundError(f"No cached dataset at {split_dir}. Provide dataset_name to download it first.")
        
        hf_dataset = load_dataset(dataset_name, split=split)
        split_dir.mkdir(parents=True, exist_ok=True)
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
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"DataLoader created — {len(data_loader.dataset)} images ({local_name}, {split})")
    
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

BDD_SIZE: Tuple[int, int] = (720, 1280)

BDD_NORMALIZATION = (
    (0.279, 0.293, 0.290),
    (0.247, 0.265, 0.276)
)

class BDDDataset(torch.utils.data.Dataset):
    """Dataset for BDD100K / BDD10K images with optional segmentation masks.

    PIL images are passed directly to ``transform``; the default
    ``make_transforms`` pipeline (``RandomResizedCrop`` -> ``ToTensor`` ->
    ``Normalize``) handles cropping, CHW conversion, and normalisation.

    Labels are loaded automatically when a sibling ``labels/`` directory
    exists next to the ``images/`` directory.  Otherwise returns
    ``(tensor, -1)``.
    """

    def __init__(
        self,
        images_dir: Union[str, Path],
        transform: Callable[..., torch.Tensor],
        patcher: Optional[Callable[..., torch.Tensor]] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.image_paths = list(self.images_dir.glob("*.jpg"))
        self.transform = transform
        self.patcher = patcher

        labels_dir = self.images_dir.parent.with_name("labels") / self.images_dir.name
        self.labels_dir = labels_dir if labels_dir.is_dir() else None

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        img_path = self.image_paths[idx]
        img = self.transform(tvio.read_image(str(img_path), mode=tvio.ImageReadMode.RGB))
        label = -1
        if self.labels_dir:
            label = tvio.read_image(str(self.labels_dir / f"{img_path.stem}_train_id.png"), mode=tvio.ImageReadMode.GRAY).squeeze(0).long()
        return (
            self.patcher(img.unsqueeze(0)).squeeze(0) if self.patcher else img,
            label
        )

def make_bdd(
    mode: str,
    split: str,
    transform: Callable[..., torch.Tensor],
    patcher: Optional[Callable[..., torch.Tensor]] = None,
    collator: Optional[Callable] = None,
    batch_size: int = 1,
    pin_mem: bool = True,
    shuffle: bool = True,
    num_workers: int = 1,
    drop_last: bool = True,
    prefetch_factor: int = 1,
) -> torch.utils.data.DataLoader:
    """Build a DataLoader for the local BDD layout.

    Expected directory structure::

        <bdd_root>/{100k|10k}/images/{split}/

    Args:
        mode: ``"pretrain"`` uses ``100k`` images only; ``"segmentation"``
            uses ``10k`` images plus ``labels/{split}/*_train_id.png``.
        split: ``"train"``, ``"validation"``, or ``"test"``.
        transform: PIL images are passed directly to this; use
            ``make_transforms`` which handles cropping, ``ToTensor``, and
            normalisation.
        patcher: Optional patchifier (e.g. tokenizer ``encode``), used only
            for pretrain.
        collator: Custom ``collate_fn`` (e.g. ``MaskCollator``).
        batch_size: Per-GPU batch size.
        pin_mem: Pin memory for faster GPU transfer.
        shuffle: Shuffle the dataset.
        num_workers: Number of DataLoader workers.
        drop_last: Drop the last incomplete batch.
        prefetch_factor: Number of batches to prefetch.

    Returns:
        Configured ``torch.utils.data.DataLoader``.
    """
    mode = mode.lower()
    if mode == "pretrain": subset = "100k"
    elif mode == "segmentation": subset = "10k"
    else: raise ValueError(f"mode must be 'pretrain' or 'segmentation'; got {mode!r}")
    
    if split not in ("train", "validation", "test"): raise ValueError(f"split must be train, validation, or test; got {split!r}")

    subset_root = Path(_DATASETS_DIR, "bdd", subset)

    data_loader = torch.utils.data.DataLoader(
        dataset=BDDDataset(
            images_dir=subset_root / "images" / split,
            transform=transform,
            patcher=patcher,
        ),
        collate_fn=collator,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"DataLoader created — {len(data_loader.dataset)} images (bdd/{subset}, {split})")
    
    return data_loader

def _label_to_rgb(label: np.ndarray) -> np.ndarray:
    """Convert a class-index mask to an RGB image for display.

    Args:
        label: Integer array of shape ``(H, W)`` with class indices.

    Returns:
        Float array of shape ``(H, W, 3)`` in ``[0, 1]``.
    """
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, color in TRAINID_TO_COLOR.items(): out[label == cid] = color
    return out.astype(np.float32) / 255.0

def _overlay_mask_on_image(
    image: torch.Tensor,
    mask_indices: torch.Tensor,
    patch_size: int,
    grid_size: Tuple[int, int],
    darken: float = 0.65,
) -> torch.Tensor:
    """Darken unmasked regions to highlight the masked patches.

    Args:
        image: Image tensor of shape ``(C, H, W)``.
        mask_indices: 1-D tensor of flattened patch indices to highlight.
        patch_size: Side length of each square patch (pixels).
        grid_size: ``(grid_h, grid_w)`` patch-grid dimensions.
        darken: Multiplicative factor applied to highlighted patches.

    Returns:
        Image tensor of shape ``(C, H, W)`` with unmasked areas zeroed out.
    """
    grid_h, grid_w = grid_size
    mask_indices = mask_indices.flatten().to(torch.long)
    mask_grid = torch.zeros((grid_h, grid_w), device=image.device, dtype=torch.float32)
    mask_grid[torch.div(mask_indices, grid_w, rounding_mode="floor"), mask_indices % grid_w] = 1.0
    mask_pixels = mask_grid.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)[: image.shape[1], : image.shape[2]]
    return image * mask_pixels * darken

def main() -> None:
    import argparse
    from models import Tokenizer
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Visualise masking on ImageNet or BDD.")
    parser.add_argument("--dataset", choices=["imagenet", "bdd"], default="imagenet")
    args = parser.parse_args()

    if args.dataset == "imagenet":
        tokenizer = Tokenizer(img_size=IMAGENET_SIZE, patch_size=16)
        collator = MaskCollator(input_size=IMAGENET_SIZE)
        data_loader = make_imagenet(
            dataset_name="timm/mini-imagenet",
            local_name="mini-imagenet",
            split="validation",
            transform=make_transforms(crop_size=IMAGENET_SIZE),
            patcher=tokenizer.encode,
            collator=collator,
        )
    else:
        tokenizer = Tokenizer(img_size=BDD_SIZE, patch_size=16)
        collator = MaskCollator(input_size=BDD_SIZE)
        data_loader = make_bdd(
            mode="pretrain",
            split="validation",
            transform=make_transforms(),
            patcher=tokenizer.encode,
            collator=collator,
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

if __name__ == "__main__": main()
