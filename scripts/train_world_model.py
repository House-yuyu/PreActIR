#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preactir.config import load_config
from preactir.data.datasets import GroupedBatchSampler, TransitionDataset
from preactir.models.factory import build_world_model
from preactir.models.losses import world_model_loss
from preactir.utils.checkpoint import save_checkpoint
from preactir.utils.io import ensure_dir
from preactir.utils.seed import make_generator, seed_everything, seed_worker
from preactir.utils.train import AverageMeter, autocast_context, make_grad_scaler, move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the tool-conditioned transition world model.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest-root", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override config seed for independent repeated training runs.",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Initialize model weights only; optimizer, scheduler, and epoch start fresh.",
    )
    parser.add_argument("--exclude-tools", nargs="*", default=[])
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help=(
            "Refit on the full training manifest for a fixed --epochs budget. "
            "In this mode best.pt is the final epoch and no early stopping is used."
        ),
    )
    return parser.parse_args()


def forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        image=batch["image"],
        belief=batch["belief"],
        action_mask=batch["action_mask"],
        tool_id=batch["tool_id"],
        target_index=batch["target_index"],
        strength=batch["strength"],
        mask_area=batch["mask_area"],
        cost_prior=batch["cost_prior"],
        trajectory_context=batch.get("trajectory_context"),
    )


def compute_metrics(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    degradation_mae = torch.abs(output["delta_degradation_mu"] - batch["delta_degradation"]).mean()
    quality_mae = torch.abs(output["delta_quality_mu"] - batch["delta_quality"]).mean()
    damage_mae = torch.abs(output["damage_mu"] - batch["damage"]).mean()
    accept_pred = (torch.sigmoid(output["accept_logit"]) >= 0.5).float()
    accept_accuracy = (accept_pred == batch["accepted"]).float().mean()
    target_pred = output["delta_degradation_mu"].gather(1, batch["target_index"][:, None]).squeeze(1)
    target_true = batch["delta_degradation"].gather(1, batch["target_index"][:, None]).squeeze(1)
    target_gain_mae = torch.abs(target_pred - target_true).mean()
    metrics = {
        "degradation_mae": float(degradation_mae.item()),
        "quality_mae": float(quality_mae.item()),
        "damage_mae": float(damage_mae.item()),
        "accept_acc": float(accept_accuracy.item()),
        "target_gain_mae": float(target_gain_mae.item()),
    }
    if "harm_logit" in output:
        harm_probability = torch.sigmoid(output["harm_logit"])
        harm_target = batch["harmful"]
        metrics.update(
            {
                "harm_acc": float(((harm_probability >= 0.5) == (harm_target >= 0.5)).float().mean().item()),
                "harm_brier": float(torch.mean((harm_probability - harm_target) ** 2).item()),
            }
        )
    if "severe_harm_logit" in output:
        severe_probability = torch.sigmoid(output["severe_harm_logit"])
        severe_target = batch["severe_harm"]
        metrics.update(
            {
                "severe_harm_acc": float(
                    (
                        (severe_probability >= 0.5)
                        == (severe_target >= 0.5)
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "severe_harm_brier": float(
                    torch.mean((severe_probability - severe_target) ** 2).item()
                ),
            }
        )
    if "paper_quality_mu" in output:
        paper_error = torch.abs(
            output["paper_quality_mu"] - batch["paper_quality_gain"]
        )
        metrics.update(
            {
                "paper_quality_mae": float(paper_error.mean().item()),
                "paper_psnr_mae_db": float((paper_error[:, 0].mean() * 10.0).item()),
                "paper_ssim_mae": float(paper_error[:, 1].mean().item()),
            }
        )
    return metrics


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp: bool,
    grad_clip: float,
    belief_noise_std: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    meter_names = [
        "loss",
        "degradation",
        "quality",
        "damage",
        "accept",
        "harm",
        "severe_harm",
        "paper_quality",
        "paper_ranking",
        "ranking",
        "degradation_mae",
        "quality_mae",
        "damage_mae",
        "accept_acc",
        "target_gain_mae",
    ]
    if bool(getattr(model, "predict_harm", False)):
        meter_names.extend(["harm_acc", "harm_brier"])
    if bool(getattr(model, "predict_severe_harm", False)):
        meter_names.extend(["severe_harm_acc", "severe_harm_brier"])
    if bool(getattr(model, "predict_paper_quality", False)):
        meter_names.extend(
            ["paper_quality_mae", "paper_psnr_mae_db", "paper_ssim_mae"]
        )
    meters = {name: AverageMeter() for name in meter_names}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            batch = move_to_device(batch, device)
            if training and belief_noise_std > 0:
                batch["belief"] = torch.clamp(
                    batch["belief"] + torch.randn_like(batch["belief"]) * belief_noise_std,
                    0.0,
                    1.0,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                output = forward_model(model, batch)
                loss, terms = world_model_loss(output, batch, weights)
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
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    run_seed = int(cfg.seed if args.seed is None else args.seed)
    seed_everything(run_seed)
    device = torch.device(args.device)
    data_root = args.data_root or cfg.data.root
    train_cfg = cfg.train_world
    epochs = args.epochs or int(train_cfg.epochs)
    exclude_tools = set(args.exclude_tools)

    train_dataset = TransitionDataset(
        data_root,
        "train",
        list(cfg.degradations.names),
        list(cfg.tools.names),
        image_size=int(cfg.data.image_size),
        augment=True,
        exclude_tools=exclude_tools,
        manifest_root=args.manifest_root,
    )
    val_dataset = None
    if not args.no_validation:
        val_dataset = TransitionDataset(
            data_root,
            "val",
            list(cfg.degradations.names),
            list(cfg.tools.names),
            image_size=int(cfg.data.image_size),
            augment=False,
            exclude_tools=exclude_tools,
            manifest_root=args.manifest_root,
        )
    loader_kwargs = dict(
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(run_seed),
        persistent_workers=int(cfg.data.num_workers) > 0,
    )
    grouped_batches = bool(train_cfg.get("grouped_batches", False))
    if grouped_batches:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=GroupedBatchSampler(
                train_dataset.ranking_group_ids,
                int(train_cfg.batch_size),
                shuffle=True,
                seed=run_seed,
            ),
            **loader_kwargs,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_sampler=GroupedBatchSampler(
                    val_dataset.ranking_group_ids,
                    int(train_cfg.batch_size),
                    shuffle=False,
                    seed=run_seed,
                ),
                **loader_kwargs,
            )
            if val_dataset is not None
            else None
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg.batch_size),
            shuffle=True,
            drop_last=False,
            **loader_kwargs,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=int(train_cfg.batch_size),
                shuffle=False,
                drop_last=False,
                **loader_kwargs,
            )
            if val_dataset is not None
            else None
        )

    model = build_world_model(cfg).to(device)
    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        incompatible = model.load_state_dict(initialization["model"], strict=False)
        allowed_missing = {
            "severe_harm_head.weight",
            "severe_harm_head.bias",
            "paper_quality_head.weight",
            "paper_quality_head.bias",
            "trajectory_projection.0.weight",
            "trajectory_projection.0.bias",
            "trajectory_projection.2.weight",
            "trajectory_projection.2.bias",
        }
        unexpected = set(incompatible.unexpected_keys)
        missing = set(incompatible.missing_keys)
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(
                "Initialization checkpoint is incompatible beyond the optional "
                f"v7 severe head: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(train_cfg.amp) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)
    start_epoch = 0
    selection_metric = str(train_cfg.get("selection_metric", "loss"))
    best_value = float("inf")
    epochs_without_improvement = 0
    patience = 0 if args.no_validation else int(train_cfg.get("early_stopping_patience", 0))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_checkpoint_path = Path(args.resume).parent / "best.pt"
        best_payload = (
            torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
            if best_checkpoint_path.is_file()
            else checkpoint
        )
        if not args.no_validation:
            best_value = float(
                best_payload.get("metrics", {}).get(f"val_{selection_metric}", best_value)
            )
        resume_history_path = Path(args.resume).parent / "history.jsonl"
        if resume_history_path.is_file() and not args.no_validation:
            history = [json.loads(line) for line in resume_history_path.read_text().splitlines() if line]
            if history:
                best_epoch = min(
                    history, key=lambda row: float(row[f"val_{selection_metric}"])
                )["epoch"]
                epochs_without_improvement = max(0, start_epoch - int(best_epoch) - 1)

    save_dir = ensure_dir(args.save_dir or train_cfg.save_dir)
    history_path = Path(save_dir) / "history.jsonl"
    weights = {key: float(value) for key, value in train_cfg.loss_weights.items()}
    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            weights,
            optimizer,
            scaler,
            amp_enabled,
            float(train_cfg.grad_clip),
            float(cfg.world_model.belief_noise_std),
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
                0.0,
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

        checkpoint_metrics = {
            **({"val_loss": val_metrics["loss"]} if val_metrics else {}),
            **record,
        }
        extra = {
            "degradation_names": list(cfg.degradations.names),
            "tool_names": list(cfg.tools.names),
            "excluded_tools": sorted(exclude_tools),
            "manifest_root": args.manifest_root,
            "predict_harm": bool(getattr(model, "predict_harm", False)),
            "predict_severe_harm": bool(
                getattr(model, "predict_severe_harm", False)
            ),
            "predict_paper_quality": bool(
                getattr(model, "predict_paper_quality", False)
            ),
            "trajectory_context_dim": int(
                getattr(model, "trajectory_context_dim", 0)
            ),
            "grouped_batches": grouped_batches,
            "seed": run_seed,
            "selection_mode": "final_epoch" if args.no_validation else "validation",
            "init_checkpoint": (
                str(Path(args.init_checkpoint).resolve())
                if args.init_checkpoint is not None
                else None
            ),
            "init_missing_keys": (
                sorted(incompatible.missing_keys)
                if args.init_checkpoint is not None
                else []
            ),
        }
        save_checkpoint(Path(save_dir) / "last.pt", model, optimizer, scheduler, epoch, checkpoint_metrics, extra)
        if args.no_validation:
            save_checkpoint(
                Path(save_dir) / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                checkpoint_metrics,
                extra,
            )
            continue
        if selection_metric not in val_metrics:
            raise KeyError(f"Unknown train_world.selection_metric={selection_metric!r}")
        if val_metrics[selection_metric] < best_value:
            best_value = val_metrics[selection_metric]
            epochs_without_improvement = 0
            save_checkpoint(Path(save_dir) / "best.pt", model, optimizer, scheduler, epoch, checkpoint_metrics, extra)
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "selection_metric": selection_metric,
                        "best_value": best_value,
                        "patience": patience,
                    }
                )
            )
            break


if __name__ == "__main__":
    main()
