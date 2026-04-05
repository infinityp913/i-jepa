import copy
import csv
import math
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from data_utils import MaskCollator, apply_masks, make_transforms, make_imagenet, make_bdd, IMAGENET_SIZE, IMAGENET_NORMALIZATION, BDD_SIZE, BDD_NORMALIZATION
from models import Encoder, Predictor, Tokenizer, ViT
from torch.nn.functional import smooth_l1_loss, mse_loss, cross_entropy

logger = logging.getLogger(__name__)

def _cosine_anneal(step, total_steps, start, end):
    """Cosine interpolation from start (step=0) to end (step=total_steps)."""
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * step / total_steps)) if total_steps > 0 else end

def run_pretrain_epoch(
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
    """Run one full I-JEPA pre-training epoch."""
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


def run_finetune_epoch(
    model,
    loader,
    device="cpu",
    train=True,
    optimizer=None,
    lr_scheduler=None,
    full_tune=False,
):
    """Run one full linear-probe fine-tuning epoch."""
    model.train(train)
    if not full_tune: model.feature_extractor.eval()

    total_loss = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for patches, labels in pbar:
            loss = cross_entropy(model(patches.to(device)), labels.to(device))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    max_norm=1.0,
                )
                optimizer.step()
                if lr_scheduler is not None: lr_scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def evaluate(model, loader, run_name=None, device="cpu"):
    """Load best checkpoint into model and evaluate accuracy on the given DataLoader."""
    if run_name is not None:
        ckpt = torch.load(
            Path("checkpoints") / run_name / "best.pt",
            map_location="cpu", 
            weights_only=True,
        )
        model.load_state_dict(ckpt["model"])
        logger.info(f"Loaded '{run_name}' (epoch {ckpt.get('epoch', '?')})")
    
    model = model.to(device).eval()
    correct, total = 0, 0
    with torch.no_grad():
        for patches, labels in tqdm(loader, desc="eval", leave=False):
            correct += (model(patches.to(device)).argmax(dim=-1) == labels.to(device)).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    logger.info(f"accuracy={accuracy:.4f} ({correct}/{total})")
    return accuracy

def trainer(
    train_loader,
    val_loader,
    task="pretrain",
    # pretrain-only
    context_encoder=None,
    predictor=None,
    ema_decay_start=0.996,
    ema_decay_end=0.999,
    # finetune-only
    model=None,
    pretrain_checkpoint=None,
    full_tune=False,
    # shared
    lr=1e-3,
    weight_decay=0.05,
    epochs=100,
    warmup_ratio=0.05,
    min_lr_ratio=0.1,
    device="cpu",
    run_name="model",
):
    logger.info(f"Task: {task} | device: {device}")

    if task == "pretrain":
        target_encoder = copy.deepcopy(context_encoder)
        for p in target_encoder.parameters(): p.requires_grad_(False)

        context_encoder = context_encoder.to(device)
        predictor = predictor.to(device)
        target_encoder = target_encoder.to(device)

        params = list(context_encoder.parameters()) + list(predictor.parameters())
    else:
        if pretrain_checkpoint is not None: model.feature_extractor.load_state_dict(torch.load(
            Path("checkpoints") / pretrain_checkpoint / "best.pt",
            map_location="cpu", 
            weights_only=True,
        )["target_encoder"])
        else: full_tune = True

        if not full_tune: 
            for p in model.feature_extractor.parameters(): p.requires_grad_(False)

        model = model.to(device)

        params = filter(lambda p: p.requires_grad, model.parameters())
        

    optimizer = torch.optim.AdamW(
        params,
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

    save_path = Path("checkpoints") / f"{run_name}_{task}"
    save_path.mkdir(parents=True, exist_ok=True)

    loss_csv = save_path / "loss.csv"
    csv_header = ["epoch", "train_loss", "val_loss"]
    with open(loss_csv, "w", newline="") as f: csv.writer(f).writerow(csv_header)

    best_val = float("inf")
    global_step = 0
    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, epochs + 1), desc="epochs"):
        if task == "pretrain":
            train_loss = run_pretrain_epoch(
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

            val_loss = run_pretrain_epoch(
                context_encoder,
                target_encoder,
                predictor,
                val_loader,
                device,
                train=False,
            )

            ckpt = {
                "context_encoder": context_encoder.state_dict(),
                "target_encoder":  target_encoder.state_dict(),
                "predictor":       predictor.state_dict(),
            }
        else:
            train_loss = run_finetune_epoch(
                model,
                train_loader,
                device,
                train=True,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            val_loss = run_finetune_epoch(
                model,
                val_loader,
                device,
                train=False,
            )

            ckpt = {"model": model.state_dict()}

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict()
                } | ckpt, 
                save_path / "best.pt"
            )

        logger.info(f"Epoch {epoch:>3}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f}")
        with open(loss_csv, "a", newline="") as f: csv.writer(f).writerow([epoch, train_loss, val_loss])
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    ep = range(1, epochs + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(ep, train_losses, label="train")
    plt.plot(ep, val_losses,   label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{task}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "loss.png")
    plt.show()
    logger.info(f"Training complete. Best val loss: {best_val:.4f}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    TASK = "finetune"

    patch_size = 16
    encoder_dim = 768
    predictor_dim = 384
    n_head = 8
    num_patches = math.prod(IMAGENET_SIZE) // patch_size ** 2

    tokenizer = Tokenizer(IMAGENET_SIZE, patch_size)

    data_loader_cfg = dict(
        local_name="mini-imagenet",
        transform=make_transforms(crop_size=IMAGENET_SIZE, normalization=IMAGENET_NORMALIZATION),
        collator=MaskCollator(input_size=IMAGENET_SIZE, patch_size=patch_size) if TASK == "pretrain" else None,
        patcher=tokenizer.encode,
        num_workers=2,
    )

    train_loader = make_imagenet(
        **data_loader_cfg,
        batch_size=64,
        split="train",
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_imagenet(
        **data_loader_cfg,
        batch_size=256,
        split="validation",
        shuffle=False,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_cfg = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        task=TASK,
        lr=1.5e-4,
        weight_decay=0.05,
        epochs=100,
        warmup_ratio=0.1,
        min_lr_ratio=1e-4,
        device=device,
    )

    if TASK == "pretrain":
        context_encoder = Encoder(
            num_patches=num_patches,
            patch_size=patch_size,
            d_model=encoder_dim,
            n_head=n_head,
            n_layers=12,
        )
        predictor = Predictor(
            num_patches=num_patches,
            embed_dim=encoder_dim,
            d_model=predictor_dim,
            n_head=n_head,
            n_layers=6,
        )

        trainer(
            **trainer_cfg,
            context_encoder=context_encoder,
            predictor=predictor,
            ema_decay_start=0.996,
            ema_decay_end=1.0,
            run_name="bdd",
        )
    else:
        test_loader = make_imagenet(
            **data_loader_cfg,
            batch_size=256,
            split="test",
            shuffle=False,
            drop_last=False,
        )
        vit = ViT(
            num_patches=num_patches,
            num_classes=100,
            patch_size=patch_size,
            d_model=encoder_dim,
            n_head=n_head,
            n_layers=12,
        )
        # trainer(
        #     **trainer_cfg,
        #     model=vit,
        #     pretrain_checkpoint="pretrain",
        #     full_tune=False,
        #     run_name="target",
        # )
        evaluate(vit, test_loader, "context_finetune", device)
        

if __name__ == "__main__": main()