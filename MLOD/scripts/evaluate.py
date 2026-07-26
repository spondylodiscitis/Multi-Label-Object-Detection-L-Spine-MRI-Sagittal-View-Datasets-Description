#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import SpineDataset, collate_fn
from src.metrics import evaluate_map
from src.models import build_model
from src.trainer import resolve_device
from utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLOD")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config.data.root = args.data_root

    device = resolve_device(config.train.device)
    dataset = SpineDataset(
        config.data.root,
        args.split,
        config.data.num_classes,
        config.data.image_size,
        min_box_wh=config.data.min_box_wh,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
    )

    model = build_model(config, pretrained_backbone=False)
    load_checkpoint(model, args.checkpoint, device, strict=True)
    model.eval()

    map50, map5095 = evaluate_map(
        model,
        loader,
        device,
        config.data.num_classes,
    )
    result = {"split": args.split, "mAP50": map50, "mAP50_95": map5095}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
