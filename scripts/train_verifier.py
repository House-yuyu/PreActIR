#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from preactir.config import load_config
from preactir.data.datasets import TransitionDataset
from preactir.models.factory import build_verifier_model
from preactir.models.losses import verifier_model_loss
from preactir.utils.checkpoint import save_checkpoint
from preactir.utils.io import ensure_dir
from preactir.utils.seed import make_generator, seed_everything, seed_worker
from preactir.utils.train import AverageMeter, autocast_context, make_grad_scaler, move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the independent before/after transition verifier.")
    parser.add_argument("--config", default="configs/preactir_small.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest-root", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
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
        help="Refit on all training identities; best.pt is the fixed-budget final epoch.",
    )
    return parser.parse_args()


def forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(
        before=batch["image"],
        after=batch["next_image"],
        action_mask=batch["action_mask"],
        tool_id=batch["tool_id"],
        target_index=batch["target_index"],
        strength=batch["strength"],
        mask_area=batch["mask_area"],
        cost_prior=batch["cost_prior"],
        trajectory_context=batch.get("trajectory_context"),
    )


def compute_metrics(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    accept_prediction = (torch.sigmoid(output["accept_logit"]) >= 0.5).float()
    status_prediction = output["status_logits"].argmax(dim=1)
    metrics = {
        "accept_acc": float((accept_prediction == batch["accepted"]).float().mean().item()),
        "status_acc": float((status_prediction == batch["status"]).float().mean().item()),
        "target_gain_mae": float(torch.abs(output["target_gain"] - batch["target_gain"]).mean().item()),
        "side_effect_mae": float(
            torch.abs(output["side_effect"] - batch["max_side_effect"]).mean().item()
        ),
        "outside_change_mae": float(
            torch.abs(output["outside_change"] - batch["outside_change"]).mean().item()
        ),
    }
    if "paper_quality_mu" in output and "paper_accept_logit" in output:
        paper_error = torch.abs(
            output["paper_quality_mu"] - batch["paper_quality_gain"]
        )
        paper_accept_prediction = (
            torch.sigmoid(output["paper_accept_logit"]) >= 0.5
        )
        metrics.update(
            {
                "paper_quality_mae": float(paper_error.mean().item()),
                "paper_psnr_mae_db": float(
                    (paper_error[:, 0].mean() * 10.0).item()
                ),
                "paper_ssim_mae": float(paper_error[:, 1].mean().item()),
                "paper_accept_acc": float(
                    (
                        paper_accept_prediction
                        == (batch["paper_accept"] >= 0.5)
                    )
                    .float()
                    .mean()
                    .item()
                ),
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
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    names = [
        "loss",
        "accept",
        "status",
        "target_gain",
        "side_effect",
        "outside_change",
        "paper_quality",
        "paper_accept",
        "accept_acc",
        "status_acc",
        "target_gain_mae",
        "side_effect_mae",
        "outside_change_mae",
    ]
    if bool(getattr(model, "predict_paper_quality", False)):
        names.extend(
            [
                "paper_quality_mae",
                "paper_psnr_mae_db",
                "paper_ssim_mae",
                "paper_accept_acc",
            ]
        )
    meters = {name: AverageMeter() for name in names}
    tp = fp = fn = tn = 0
    paper_tp = paper_fp = paper_fn = paper_tn = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            batch = move_to_device(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                output = forward_model(model, batch)
                loss, terms = verifier_model_loss(output, batch, weights)
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
            accept_prediction = torch.sigmoid(output["accept_logit"]) >= 0.5
            accept_target = batch["accepted"] >= 0.5
            tp += int(torch.logical_and(accept_prediction, accept_target).sum().item())
            fp += int(torch.logical_and(accept_prediction, ~accept_target).sum().item())
            fn += int(torch.logical_and(~accept_prediction, accept_target).sum().item())
            tn += int(torch.logical_and(~accept_prediction, ~accept_target).sum().item())
            if "paper_accept_logit" in output:
                paper_prediction = (
                    torch.sigmoid(output["paper_accept_logit"]) >= 0.5
                )
                paper_target = batch["paper_accept"] >= 0.5
                paper_tp += int(
                    torch.logical_and(paper_prediction, paper_target).sum().item()
                )
                paper_fp += int(
                    torch.logical_and(paper_prediction, ~paper_target).sum().item()
                )
                paper_fn += int(
                    torch.logical_and(~paper_prediction, paper_target).sum().item()
                )
                paper_tn += int(
                    torch.logical_and(~paper_prediction, ~paper_target).sum().item()
                )
    result = {name: meter.average for name, meter in meters.items()}
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta_sq = 0.25
    result.update(
        {
            "accept_precision": precision,
            "accept_recall": recall,
            "accept_f0_5": (1.0 + beta_sq)
            * precision
            * recall
            / max(beta_sq * precision + recall, 1e-12),
            "accept_tp": float(tp),
            "accept_fp": float(fp),
            "accept_fn": float(fn),
            "accept_tn": float(tn),
        }
    )
    if bool(getattr(model, "predict_paper_quality", False)):
        paper_precision = paper_tp / max(paper_tp + paper_fp, 1)
        paper_recall = paper_tp / max(paper_tp + paper_fn, 1)
        result.update(
            {
                "paper_accept_precision": paper_precision,
                "paper_accept_recall": paper_recall,
                "paper_accept_f0_5": (1.0 + beta_sq)
                * paper_precision
                * paper_recall
                / max(beta_sq * paper_precision + paper_recall, 1e-12),
                "paper_accept_tp": float(paper_tp),
                "paper_accept_fp": float(paper_fp),
                "paper_accept_fn": float(paper_fn),
                "paper_accept_tn": float(paper_tn),
            }
        )
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    run_seed = int(cfg.seed if args.seed is None else args.seed)
    seed_everything(run_seed)
    device = torch.device(args.device)
    data_root = args.data_root or cfg.data.root
    train_cfg = cfg.train_verifier
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
        batch_size=int(train_cfg.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=make_generator(run_seed),
        persistent_workers=int(cfg.data.num_workers) > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = (
        DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
        if val_dataset is not None
        else None
    )

    model = build_verifier_model(cfg).to(device)
    if args.init_checkpoint:
        initialization = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        incompatible = model.load_state_dict(initialization["model"], strict=False)
        allowed_missing = {
            "paper_quality_head.weight",
            "paper_quality_head.bias",
            "paper_accept_head.weight",
            "paper_accept_head.bias",
            "trajectory_projection.0.weight",
            "trajectory_projection.0.bias",
            "trajectory_projection.2.weight",
            "trajectory_projection.2.bias",
        }
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(
                "Verifier initialization checkpoint is incompatible: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(train_cfg.lr), weight_decay=float(train_cfg.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp_enabled = bool(train_cfg.amp) and device.type == "cuda"
    scaler = make_grad_scaler(device, amp_enabled)
    start_epoch = 0
    selection_metric = str(train_cfg.get("selection_metric", "accept_f0_5"))
    selection_mode = str(train_cfg.get("selection_mode", "max"))
    if selection_mode not in {"min", "max"}:
        raise ValueError("train_verifier.selection_mode must be min or max")
    best_value = float("inf") if selection_mode == "min" else -float("inf")
    epochs_without_improvement = 0
    patience = 0 if args.no_validation else int(
        train_cfg.get("early_stopping_patience", 0)
    )
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if not args.no_validation:
            best_value = float(
                checkpoint.get("metrics", {}).get(
                    f"val_{selection_metric}", best_value
                )
            )

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
            "tool_names": list(cfg.tools.names),
            "excluded_tools": sorted(exclude_tools),
            "manifest_root": args.manifest_root,
            "selection_mode": "final_epoch" if args.no_validation else "validation",
            "selection_metric": selection_metric,
            "selection_direction": selection_mode,
            "predict_paper_quality": bool(
                getattr(model, "predict_paper_quality", False)
            ),
            "trajectory_context_dim": int(
                getattr(model, "trajectory_context_dim", 0)
            ),
            "seed": run_seed,
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
        if selection_metric not in val_metrics:
            raise KeyError(f"Unknown train_verifier.selection_metric={selection_metric!r}")
        value = float(val_metrics[selection_metric])
        improved = value < best_value if selection_mode == "min" else value > best_value
        if improved:
            best_value = value
            epochs_without_improvement = 0
            save_checkpoint(Path(save_dir) / "best.pt", model, optimizer, scheduler, epoch, metrics, extra)
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "selection_metric": selection_metric,
                        "selection_direction": selection_mode,
                        "best_value": best_value,
                        "patience": patience,
                    }
                )
            )
            break


if __name__ == "__main__":
    main()
