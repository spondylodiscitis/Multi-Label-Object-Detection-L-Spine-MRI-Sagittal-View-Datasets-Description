from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch


def save_checkpoint_atomic(state: dict, path: str | Path) -> bool:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")

    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
        return True
    except Exception as error:
        print(f"[WARN] Checkpoint save failed: {destination}: {error}")
        temporary.unlink(missing_ok=True)
        return False


def extract_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint
    raise RuntimeError("Checkpoint does not contain a supported state_dict.")


def load_checkpoint(
    model,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = True,
):
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)

    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    return checkpoint
