from __future__ import annotations

import numpy as np
import torch


def sanitize_boxes_and_labels(
    boxes: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    min_wh: float = 2.0,
):
    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, labels.shape[1]), dtype=np.float32),
        )

    boxes = boxes.astype(np.float32, copy=True)
    labels = labels.astype(np.float32, copy=True)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_size - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_size - 1)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep = (
        np.isfinite(boxes).all(axis=1)
        & np.isfinite(labels).all(axis=1)
        & (widths >= min_wh)
        & (heights >= min_wh)
        & (labels.sum(axis=1) >= 1)
    )
    return boxes[keep], labels[keep]


def encode_boxes(
    ground_truth: torch.Tensor,
    proposals: torch.Tensor,
) -> torch.Tensor:
    proposal_x = (proposals[:, 0] + proposals[:, 2]) * 0.5
    proposal_y = (proposals[:, 1] + proposals[:, 3]) * 0.5
    proposal_w = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    proposal_h = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)

    target_x = (ground_truth[:, 0] + ground_truth[:, 2]) * 0.5
    target_y = (ground_truth[:, 1] + ground_truth[:, 3]) * 0.5
    target_w = (ground_truth[:, 2] - ground_truth[:, 0]).clamp(min=1e-6)
    target_h = (ground_truth[:, 3] - ground_truth[:, 1]).clamp(min=1e-6)

    output = torch.stack(
        [
            (target_x - proposal_x) / proposal_w,
            (target_y - proposal_y) / proposal_h,
            torch.log(target_w / proposal_w),
            torch.log(target_h / proposal_h),
        ],
        dim=1,
    )
    return torch.nan_to_num(output)


def decode_boxes(
    deltas: torch.Tensor,
    proposals: torch.Tensor,
) -> torch.Tensor:
    proposal_x = (proposals[:, 0] + proposals[:, 2]) * 0.5
    proposal_y = (proposals[:, 1] + proposals[:, 3]) * 0.5
    proposal_w = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    proposal_h = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)

    dx, dy = deltas[:, 0], deltas[:, 1]
    dw = deltas[:, 2].clamp(-10, 10)
    dh = deltas[:, 3].clamp(-10, 10)

    center_x = proposal_x + dx * proposal_w
    center_y = proposal_y + dy * proposal_h
    width = proposal_w * torch.exp(dw)
    height = proposal_h * torch.exp(dh)

    output = torch.stack(
        [
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ],
        dim=1,
    )
    return torch.nan_to_num(output)
