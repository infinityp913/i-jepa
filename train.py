import copy
import csv
import math
import logging
from itertools import islice
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from data_utils import MaskCollator, apply_masks, make_transforms, make_imagenet, make_bdd, IMAGENET_SIZE, IMAGENET_NORMALIZATION, BDD_SIZE, BDD_NORMALIZATION
from models import Encoder, Predictor, Tokenizer, ViT
from torch.nn.functional import smooth_l1_loss, mse_loss, cross_entropy
import contextlib
from torch.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

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
    grad_accum_steps=1,
    ema_start=0.996,
    ema_end=1.0,
    step_offset=0,
    total_steps=1,
    _tgt_stream=None,
    scaler=None,
):
    """Run one full I-JEPA pre-training epoch."""
    context_encoder.train(train)
    predictor.train(train)
    target_encoder.eval()

    total_loss = torch.zeros(1, device=device)
    steps = 0
    stream_ctx = torch.cuda.stream(_tgt_stream) if _tgt_stream else contextlib.nullcontext()

    if train:
        pbar_total = (len(loader) // grad_accum_steps) * grad_accum_steps
        batch_iter = islice(loader, pbar_total)
        micro_step = 0
        opt_params = [p for group in optimizer.param_groups for p in group["params"]]
        optimizer.zero_grad(set_to_none=True)
    else:
        pbar_total = len(loader)
        batch_iter = loader

    with torch.set_grad_enabled(train):
        for (patches, _), enc_masks, pred_masks in tqdm(batch_iter, desc="train" if train else "val", leave=False, total=pbar_total):
            n_enc_masks, n_pred_masks = enc_masks.shape[0], pred_masks.shape[0]

            patches = patches.to(device, non_blocking=True)
            enc_masks = enc_masks.to(device, non_blocking=True)
            pred_masks = pred_masks.to(device, non_blocking=True)

            if _tgt_stream: _tgt_stream.wait_stream(torch.cuda.current_stream())

            with autocast(device_type=device, dtype=AMP_DTYPE):
                with stream_ctx, torch.no_grad(): 
                    y_embed = apply_masks(
                        target_encoder(patches), 
                        pred_masks
                    ).repeat(1, n_enc_masks, 1, 1).flatten(0, 1)

                x_embed = context_encoder(
                    apply_masks(patches, enc_masks).flatten(0, 1),
                    enc_masks.flatten(0, 1)
                )

                y_pred = predictor(
                    x_embed.repeat(n_pred_masks, 1, 1),
                    enc_masks.repeat(n_pred_masks, 1, 1).flatten(0, 1),
                    pred_masks.repeat(1, n_enc_masks, 1).flatten(0, 1),
                )

                if _tgt_stream: torch.cuda.current_stream().wait_stream(_tgt_stream)

                loss = smooth_l1_loss(y_pred, y_embed)
                #replace with loss from paper (below) if needed, the above loss is more stable
                #loss = mse_loss(y_pred, y_embed, reduction='sum') / y_embed.shape[0]

            if train:
                scaler.scale(loss / grad_accum_steps).backward()
                micro_step += 1

                if micro_step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if lr_scheduler is not None: lr_scheduler.step()

                    with torch.no_grad():
                        ema_weight = 1 - _cosine_anneal(
                            step_offset + micro_step // grad_accum_steps - 1,
                            total_steps - 1,
                            ema_start,
                            ema_end,
                        )
                        torch._foreach_lerp_(list(target_encoder.parameters()), list(context_encoder.parameters()), ema_weight)

            total_loss += loss.detach()
            steps += 1

    return (total_loss / steps).item()


def run_finetune_epoch(
    model,
    loader,
    device="cpu",
    train=True,
    optimizer=None,
    lr_scheduler=None,
    grad_accum_steps=1,
    full_tune=False,
    scaler=None,
):
    """Run one full linear-probe fine-tuning epoch."""
    model.train(train)
    if train and not full_tune: model.feature_extractor.eval()

    total_loss = torch.zeros(1, device=device)
    steps = 0

    if train:
        pbar_total = (len(loader) // grad_accum_steps) * grad_accum_steps
        batch_iter = islice(loader, pbar_total)
        micro_step = 0
        opt_params = [p for group in optimizer.param_groups for p in group["params"]]
        optimizer.zero_grad(set_to_none=True)
    else:
        batch_iter = loader
        pbar_total = len(loader)

    with torch.set_grad_enabled(train):
        for patches, labels in tqdm(batch_iter, desc="train" if train else "val", leave=False, total=pbar_total):
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device, dtype=AMP_DTYPE): loss = cross_entropy(model(patches), labels)

            if train:
                scaler.scale(loss / grad_accum_steps).backward()
                micro_step += 1

                if micro_step % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    if lr_scheduler is not None: lr_scheduler.step()

            total_loss += loss.detach()
            steps += 1

    return (total_loss / steps).item()


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
    correct_tensor = torch.zeros(1, dtype=torch.long, device=device)
    total = 0
    
    with torch.inference_mode():
        with autocast(device_type=device, dtype=AMP_DTYPE):
            for patches, labels in tqdm(loader, desc="eval", leave=False):
                correct_tensor += (model(patches.to(device, non_blocking=True)).argmax(dim=-1) == labels.to(device, non_blocking=True)).sum()
                total += labels.size(0)
    
    accuracy = correct_tensor.item() / total
    logger.info(f"accuracy={accuracy:.4f} ({correct_tensor.item()}/{total})")
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
    grad_accum_steps=1,
    warmup_ratio=0.05,
    min_lr_ratio=0.1,
    device="cpu",
    run_name="model",
):
    n_full = len(train_loader)
    if n_full < grad_accum_steps: raise ValueError(f"len(train_loader)={n_full} must be >= grad_accum_steps={grad_accum_steps}")
    dropped = n_full % grad_accum_steps
    opt_steps_per_epoch = n_full // grad_accum_steps
    if dropped: logger.warning(f"Each train epoch skips the last {dropped} batches (incomplete gradient accumulation).")

    logger.info(
        f"Task: {task} | device: {device} | grad_accum_steps={grad_accum_steps} "
        f"({opt_steps_per_epoch} optimizer steps / epoch)"
    )

    if task == "pretrain":
        target_encoder = copy.deepcopy(context_encoder)
        for p in target_encoder.parameters(): p.requires_grad_(False)

        context_encoder = torch.compile(context_encoder.to(device), backend="aot_eager")
        predictor = torch.compile(predictor.to(device), backend="aot_eager")
        target_encoder = torch.compile(target_encoder.to(device), backend="aot_eager")

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

        model = torch.compile(model.to(device), backend="aot_eager")

        params = filter(lambda p: p.requires_grad, model.parameters())
        

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
        fused=True
    )

    total_steps = epochs * opt_steps_per_epoch
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
    global_opt_step = 0
    train_losses, val_losses = [], []

    if task == "pretrain": _tgt_stream = torch.cuda.Stream(device) if device != "cpu" else None
    scaler = GradScaler(device, enabled=(AMP_DTYPE == torch.float16))

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
                grad_accum_steps=grad_accum_steps,
                ema_start=ema_decay_start,
                ema_end=ema_decay_end,
                step_offset=global_opt_step,
                total_steps=total_steps,
                _tgt_stream=_tgt_stream,
                scaler=scaler,
            )

            global_opt_step += opt_steps_per_epoch

            val_loss = run_pretrain_epoch(
                context_encoder,
                target_encoder,
                predictor,
                val_loader,
                device,
                train=False,
                _tgt_stream=_tgt_stream,
                scaler=scaler,
            )

        else:
            train_loss = run_finetune_epoch(
                model,
                train_loader,
                device,
                train=True,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_accum_steps=grad_accum_steps,
                full_tune=full_tune,
                scaler=scaler,
            )
            val_loss = run_finetune_epoch(
                model,
                val_loader,
                device,
                train=False,
                scaler=scaler,
            )

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "context_encoder": context_encoder.state_dict(),
                "target_encoder":  target_encoder.state_dict(),
                "predictor":       predictor.state_dict(),
            } if task == "pretrain" else {"model": model.state_dict()}
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
    logger.info(f"AMP_DTYPE: {AMP_DTYPE}")

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
        prefetch_factor=2,
    )

    train_loader = make_imagenet(
        **data_loader_cfg,
        batch_size=128,
        split="train",
        shuffle=True,
        drop_last=True,
    )
    val_loader = make_imagenet(
        **data_loader_cfg,
        batch_size=128,
        split="validation",
        shuffle=False,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_cfg = dict(
        train_loader=train_loader,
        val_loader=val_loader,
        task=TASK,
        lr=1e-3,
        weight_decay=0.05,
        epochs=20,
        grad_accum_steps=4,
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
            run_name="imagenet",
        )
    else:
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
        
        test_loader = make_imagenet(
            **data_loader_cfg,
            batch_size=2048,
            split="test",
            shuffle=False,
            drop_last=False,
        )

        evaluate(vit, test_loader, "target_finetune", device)
        

if __name__ == "__main__": main()