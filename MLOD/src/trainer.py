from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import SpineDataset, build_train_transform, collate_fn
from src.metrics import evaluate_map
from src.models import build_model
from utils.checkpoint import save_checkpoint_atomic
from utils.logging import append_csv_row
from utils.reproducibility import set_seed


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def move_targets(targets, device):
    for target in targets:
        target["boxes"] = target["boxes"].to(device)
        target["labels"] = target["labels"].to(device)
    return targets


def train(config) -> None:
    set_seed(int(config.project.seed))
    device = resolve_device(config.train.device)
    print(f"[DEVICE] {device}")

    train_dataset = SpineDataset(
        config.data.root,
        "train",
        config.data.num_classes,
        config.data.image_size,
        transform=build_train_transform(),
        min_box_wh=config.data.min_box_wh,
    )
    validation_dataset = SpineDataset(
        config.data.root,
        "val",
        config.data.num_classes,
        config.data.image_size,
        transform=None,
        min_box_wh=config.data.min_box_wh,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        num_workers=int(config.train.num_workers),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=False,
        num_workers=int(config.train.num_workers),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config, pretrained_backbone=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )

    best_map50 = -1.0
    for epoch in range(int(config.train.epochs)):
        model.train()
        total_loss = 0.0
        completed_steps = 0

        progress = tqdm(
            train_loader,
            desc=f"Train {epoch + 1}/{config.train.epochs}",
            dynamic_ncols=True,
        )
        for images, targets in progress:
            images = [image.to(device) for image in images]
            targets = move_targets(targets, device)

            losses = model(images, targets)
            total = sum(losses.values())
            if not torch.isfinite(total):
                print(f"[WARN] Skipping non-finite loss: {float(total)}")
                continue

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config.train.max_gradient_norm),
            )
            optimizer.step()

            total_loss += float(total.item())
            completed_steps += 1
            progress.set_postfix(loss=f"{total.item():.4f}")

        train_loss = total_loss / max(completed_steps, 1)

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(
                validation_loader,
                desc="Validation loss",
                dynamic_ncols=True,
            ):
                images = [image.to(device) for image in images]
                targets = move_targets(targets, device)
                validation_loss += float(sum(model(images, targets).values()).item())
        validation_loss /= max(len(validation_loader), 1)

        map50, map5095 = evaluate_map(
            model,
            validation_loader,
            device,
            int(config.data.num_classes),
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "val_map50": map50,
            "val_map5095": map5095,
        }
        append_csv_row(config.train.log_csv, row)
        print(row)

        if map50 > best_map50:
            best_map50 = map50
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_map50": best_map50,
                    "last_val_loss": validation_loss,
                    "last_val_map5095": map5095,
                    "config": dict(config),
                },
                config.train.best_checkpoint,
            )

        if config.train.last_checkpoint:
            save_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_map50": best_map50,
                    "last_val_loss": validation_loss,
                    "last_val_map50": map50,
                    "last_val_map5095": map5095,
                },
                config.train.last_checkpoint,
            )
