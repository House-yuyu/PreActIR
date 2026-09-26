#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preactir.config import load_config
from preactir.data.datasets import BeliefDataset
from preactir.models.factory import build_belief_model
from preactir.models.losses import belief_loss
from preactir.utils.checkpoint import load_checkpoint, save_checkpoint
from preactir.utils.io import ensure_dir
from preactir.utils.seed import make_generator, seed_everything, seed_worker
from preactir.utils.train import AverageMeter, autocast_context, make_grad_scaler, move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the spatial degradation-belief encoder.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Override belief encoder width; the same value is required at inference.",
    )
    parser.add_argument(
        "--global-backbone",
        choices=["none", "clip_vit_b32"],
        default=None,
    )
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument(
        "--clip-fusion-mode", choices=["concat", "replace", "linear"], default=None
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--presence-pos-weight", type=float, default=None)
    parser.add_argument("--presence-focal-gamma", type=float, default=None)
    parser.add_argument("--presence-loss-weight", type=float, default=None)
    parser.add_argument("--severity-loss-weight", type=float, default=None)
    parser.add_argument("--mask-bce-loss-weight", type=float, default=None)
    parser.add_argument("--mask-dice-loss-weight", type=float, default=None)
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Refit on all training identities; best.pt is the fixed-budget final epoch.",
    )
    return parser.parse_args()


def compute_metrics(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    presence_prob = torch.sigmoid(output["presence_logits"])
    presence_pred = presence_prob >= 0.5
    presence_true = batch["presence"] >= 0.5
    presence_acc = (presence_pred == presence_true).float().mean()
    tp = (presence_pred & presence_true).float().sum()
    fp = (presence_pred & ~presence_true).float().sum()
    fn = (~presence_pred & presence_true).float().sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    presence_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)

    active = batch["presence"] > 0.5
    severity_abs = torch.abs(output["severity_mu"] - batch["severity"])
    severity_mae = severity_abs[active].mean() if bool(active.any()) else severity_abs.mean() * 0.0

    mask_pred = torch.sigmoid(output["mask_logits"]) >= 0.5
    mask_true = batch["masks"] >= 0.5
    intersection = (mask_pred & mask_true).float().sum(dim=(2, 3))
    union = (mask_pred | mask_true).float().sum(dim=(2, 3)).clamp_min(1.0)
    mask_iou = intersection / union
    mask_iou = mask_iou[active].mean() if bool(active.any()) else mask_iou.mean() * 0.0
    return {
        "presence_acc": float(presence_acc.item()),
        "presence_precision": float(precision.item()),
        "presence_recall": float(recall.item()),
        "presence_f1": float(presence_f1.item()),
        "severity_mae": float(severity_mae.item()),
        "mask_iou": float(mask_iou.item()),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp: bool,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    names = [
        "loss",
        "presence",
        "severity",
        "mask_bce",
        "mask_dice",
        "presence_acc",
        "presence_precision",
        "presence_recall",
        "presence_f1",
        "severity_mae",
        "mask_iou",
    ]
    meters = {name: AverageMeter() for name in names}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            batch = move_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                output = model(batch["image"])
                loss, terms = belief_loss(output, batch, weights)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            batch_size = int(batch["image"].shape[0])
            meters["loss"].update(float(loss.detach().item()), batch_size)
            for name, value in terms.items():
                meters[name].update(float(value.detach().item()), batch_size)
            for name, value in compute_metrics(output, batch).items():
                meters[name].update(value, batch_size)
    return {name: meter.average for name, meter in meters.items()}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.base_channels is not None:
        cfg.belief_model.base_channels = int(args.base_channels)
    if args.global_backbone is not None:
        cfg.belief_model["global_backbone"] = args.global_backbone
    if args.clip_checkpoint is not None:
        cfg.belief_model["clip_checkpoint"] = args.clip_checkpoint
    if args.clip_fusion_mode is not None:
        cfg.belief_model["clip_fusion_mode"] = args.clip_fusion_mode
    seed_everything(int(cfg.seed))
    device = torch.device(args.device)
    data_root = args.data_root or cfg.data.root
    train_cfg = cfg.train_belief
    epochs = int(args.epochs if args.epochs is not None else train_cfg.epochs)

    train_dataset = BeliefDataset(
        data_root,
        "train",
        list(cfg.degradations.names),
        image_size=int(cfg.data.image_size),
        augment=True,
    )
    val_dataset = None
    if not args.no_validation:
        val_dataset = BeliefDataset(
            data_root,
            "val",
            list(cfg.degradations.names),
            image_size=int(cfg.data.image_size),
            augment=False,
        )
    loader_kwargs = dict(
        batch_size=int(args.batch_size if args.batch_size is not None else train_cfg.batch_size),
        num_workers=int(args.num_workers if args.num_workers is not None else cfg.data.num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(int(cfg.seed)),
        persistent_workers=int(
            args.num_workers if args.num_workers is not None else cfg.data.num_workers
        )
        > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = (
        DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
        if val_dataset is not None
        else None
    )

    model = build_belief_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate if args.learning_rate is not None else train_cfg.lr),
        weight_decay=float(
            args.weight_decay if args.weight_decay is not None else train_cfg.weight_decay
        ),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(train_cfg.amp) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)
    start_epoch = 0
    best_score = float("-inf")
    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if not args.no_validation:
            best_score = float(
                checkpoint.get("metrics", {}).get("val_presence_f1", best_score)
            )

    save_dir = ensure_dir(args.save_dir or train_cfg.save_dir)
    history_path = Path(save_dir) / "history.jsonl"
    weights = {key: float(value) for key, value in train_cfg.loss_weights.items()}
    if args.presence_pos_weight is not None:
        weights["presence_pos_weight"] = float(args.presence_pos_weight)
    if args.presence_focal_gamma is not None:
        weights["presence_focal_gamma"] = float(args.presence_focal_gamma)
    if args.presence_loss_weight is not None:
        weights["presence"] = float(args.presence_loss_weight)
    if args.severity_loss_weight is not None:
        weights["severity"] = float(args.severity_loss_weight)
    if args.mask_bce_loss_weight is not None:
        weights["mask_bce"] = float(args.mask_bce_loss_weight)
    if args.mask_dice_loss_weight is not None:
        weights["mask_dice"] = float(args.mask_dice_loss_weight)
    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(
            model, train_loader, device, weights, optimizer, scaler, amp_enabled, float(train_cfg.grad_clip)
        )
        val_metrics = (
            run_epoch(
                model,
                val_loader,
                device,
                weights,
                None,
                scaler,
                amp_enabled,
                float(train_cfg.grad_clip),
            )
            if val_loader is not None
            else {}
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))

        metrics = {
            **({"val_loss": val_metrics["loss"]} if val_metrics else {}),
            **record,
        }
        extra = {
            "degradation_names": list(cfg.degradations.names),
            "selection_mode": "final_epoch" if args.no_validation else "validation",
            "base_channels": int(cfg.belief_model.base_channels),
            "global_backbone": str(cfg.belief_model.get("global_backbone", "none")),
            "clip_checkpoint": cfg.belief_model.get("clip_checkpoint"),
            "clip_fusion_mode": str(cfg.belief_model.get("clip_fusion_mode", "concat")),
            "batch_size": int(
                args.batch_size if args.batch_size is not None else train_cfg.batch_size
            ),
            "loss_weights": weights,
            "learning_rate": float(
                args.learning_rate if args.learning_rate is not None else train_cfg.lr
            ),
            "weight_decay": float(
                args.weight_decay if args.weight_decay is not None else train_cfg.weight_decay
            ),
        }
        save_checkpoint(Path(save_dir) / "last.pt", model, optimizer, scheduler, epoch, metrics, extra)
        if args.no_validation:
            save_checkpoint(
                Path(save_dir) / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                extra,
            )
            continue
        if val_metrics["presence_f1"] > best_score:
            best_score = val_metrics["presence_f1"]
            save_checkpoint(Path(save_dir) / "best.pt", model, optimizer, scheduler, epoch, metrics, extra)


if __name__ == "__main__":
    main()
