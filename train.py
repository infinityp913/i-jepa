import copy
import csv
import math
import logging
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from data_utils import MaskCollator, apply_masks, make_imagenet, make_transforms
from models import Encoder, Predictor, Tokenizer
from torch.nn.functional import smooth_l1_loss, mse_loss

logger = logging.getLogger(__name__)

def _cosine_anneal(step, total_steps, start, end):
    """Cosine interpolation from start (step=0) to end (step=total_steps)."""
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * step / total_steps)) if total_steps > 0 else end

def run_epoch(
    context_encoder,
    target_encoder,
    predictor,
    loader,
    device="cpu",
    train=True,
    optimizer=None,
    lr_scheduler=None,
    ema_start=0.996,
    ema_end=1.0,
    step_offset=0,
    total_steps=1,
):
    """Run one full epoch."""
    context_encoder.train(train)
    predictor.train(train)
    target_encoder.eval()

    total_loss = 0.0
    steps = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for (patches, _), enc_masks, pred_masks in pbar:
            patches = patches.to(device)
            enc_masks = enc_masks.to(device)
            pred_masks = pred_masks.to(device)

            x_embed = context_encoder(apply_masks(patches, enc_masks), enc_masks.flatten(0, 1))

            with torch.no_grad(): y_embed = apply_masks(target_encoder(patches), pred_masks)

            y_pred = predictor(
                x_embed.repeat(len(pred_masks), 1, 1),
                enc_masks.repeat(len(pred_masks), 1, 1).flatten(0, 1),
                pred_masks.flatten(0, 1),
            )

            loss = smooth_l1_loss(y_pred, y_embed)
            #replace with loss from paper (below) if needed, the above loss is more stable
            #loss = mse_loss(y_pred, y_embed, reduction='sum') / math.prod(pred_masks.shape[:-1])

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    max_norm=1.0,
                )
                optimizer.step()
                if lr_scheduler is not None: lr_scheduler.step()
                
                with torch.no_grad():
                    for p_ctx, p_tgt in zip(context_encoder.parameters(), target_encoder.parameters()): 
                        p_tgt.lerp_(p_ctx, 1 - _cosine_anneal(step_offset + steps, total_steps - 1, ema_start, ema_end))

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / steps

def trainer(
    train_loader,
    val_loader,
    context_encoder,
    predictor,
    lr=1e-3,
    weight_decay=0.05,
    ema_decay_start=0.996,
    ema_decay_end=0.999,
    epochs=100,
    warmup_ratio=0.05,
    min_lr_ratio=0.1,
    device="cpu",
    save_path="checkpoints",
):
    logger.info(f"Using device: {device}")

    target_encoder = copy.deepcopy(context_encoder)
    for p in target_encoder.parameters(): p.requires_grad_(False)

    context_encoder = context_encoder.to(device)
    predictor = predictor.to(device)
    target_encoder = target_encoder.to(device)

    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: ((step + 1) / warmup_steps) if step < warmup_steps 
        else _cosine_anneal(step - warmup_steps, total_steps - warmup_steps - 1, 1.0, min_lr_ratio)
    )
    logger.info(f"total_steps={total_steps}, warmup_steps={warmup_steps}")

    os.makedirs(save_path, exist_ok=True)
    best_val = float("inf")
    train_losses, val_losses = [], []
    global_step = 0

    loss_csv = os.path.join(save_path, "loss.csv")
    with open(loss_csv, "w", newline="") as f: csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        train_loss = run_epoch(
            context_encoder,
            target_encoder,
            predictor,
            train_loader,
            device,
            train=True,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_start=ema_decay_start,
            ema_end=ema_decay_end,
            step_offset=global_step,
            total_steps=total_steps,
        )
        global_step += steps_per_epoch

        val_loss = run_epoch(
            context_encoder,
            target_encoder,
            predictor,
            val_loader,
            device,
            train=False,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        with open(loss_csv, "a", newline="") as f: csv.writer(f).writerow([epoch, train_loss, val_loss])

        logger.info(f"Epoch {epoch:>3}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "context_encoder": context_encoder.state_dict(),
                "target_encoder":  target_encoder.state_dict(),
                "predictor":       predictor.state_dict(),
                "optimizer":       optimizer.state_dict(),
            }, os.path.join(save_path, "best.pt"))

    ep = range(1, epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ep, train_losses, label="train")
    plt.plot(ep, val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("I-JEPA Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.show()
    logger.info(f"Loss plot saved to {save_path}/loss.png")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    img_size = 224
    patch_size = 16
    encoder_dim = 768
    predictor_dim = 384
    n_head = 8
    # batch_size = 64
    batch_size = 2
    num_patches = (img_size // patch_size) ** 2

    tokenizer = Tokenizer(img_size, patch_size)
    context_encoder = Encoder(num_patches, patch_size=patch_size, d_model=encoder_dim, n_head=n_head, n_layers=12)
    predictor = Predictor(num_patches, encoder_dim, predictor_dim, n_head, n_layers=6)
    
    data_loader_cfg = dict(
        transform=make_transforms(crop_size=img_size),
        collator=MaskCollator(input_size=(img_size, img_size), patch_size=patch_size),
        batch_size=batch_size,
        local_name="mini-imagenet",
        patcher=tokenizer.encode,
        num_workers=2,
    )

    train_loader = make_imagenet(
        **data_loader_cfg,
        # dataset_name="timm/mini-imagenet", # only specify the first time to download
        split="train",
        shuffle=True,
        drop_last=True,
    )
    val_loader   = make_imagenet(
        **data_loader_cfg,
        # dataset_name="timm/mini-imagenet", # only specify the first time to download
        split="validation",
        shuffle=False,
        drop_last=False,
    )

    trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        context_encoder=context_encoder,
        predictor=predictor,
        lr=1e-4,
        weight_decay=0.05,
        epochs=100,
        warmup_ratio=0.1,
        min_lr_ratio=1e-4,
        ema_decay_start=0.996,
        ema_decay_end=0.999,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

if __name__ == "__main__": main()
