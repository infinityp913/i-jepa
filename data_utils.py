import math
import logging
from multiprocessing import Value
import os
import torch
import torchvision.transforms as transforms
from datasets import load_dataset, load_from_disk


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
    logger.info("Building ImageNet data transforms")

    return transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=crop_scale),
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
            torch.utils.data.default_collate([
                [cm[:min_keep_enc] for cm in cm_list] 
                for cm_list in collated_masks_enc
            ]), 
            torch.utils.data.default_collate([
                [cm[:min_keep_pred] for cm in cm_list] 
                for cm_list in collated_masks_pred
            ])
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

def _overlay_mask_on_image(image, mask_indices, patch_size, grid_size, darken=0.65):
    """Overlay a mask on an image, only for visualization."""
    grid_h, grid_w = grid_size
    mask_indices = mask_indices.flatten().to(torch.long)
    mask_grid = torch.zeros((grid_h, grid_w), device=image.device, dtype=torch.float32)
    mask_grid[torch.div(mask_indices, grid_w, rounding_mode="floor"), mask_indices % grid_w] = 1.0
    mask_pixels = mask_grid.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)[: image.shape[1], : image.shape[2]]
    return image * mask_pixels * darken

def main():
    from vit import Tokenizer
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

if __name__ == "__main__": main()