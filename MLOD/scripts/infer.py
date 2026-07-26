#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.inference import run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLOD inference")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "default.yaml"),
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/inference/result.jpg")
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.device:
        config.train.device = args.device

    run_inference(
        config=config,
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        threshold_json=args.thresholds,
        show=args.show,
    )


if __name__ == "__main__":
    main()
