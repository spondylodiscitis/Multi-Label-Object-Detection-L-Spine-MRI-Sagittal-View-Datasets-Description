from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from src.data import collate_fn


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if precisions.size == 0:
        return 0.0
    return float(
        sum(
            (
                np.max(precisions[recalls >= threshold])
                if np.any(recalls >= threshold)
                else 0.0
            )
            / 11.0
            for threshold in np.linspace(0, 1, 11)
        )
    )


def evaluate_map(
    model,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float]:
    model.eval()
    evaluation_loader = DataLoader(
        data_loader.dataset,
        batch_size=data_loader.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    detections, ground_truth = [], []
    with torch.inference_mode():
        for images, targets in tqdm(
            evaluation_loader,
            desc="Evaluation",
            dynamic_ncols=True,
        ):
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                detections.append(
                    {
                        key: value.detach().cpu()
                        for key, value in output.items()
                    }
                )
                ground_truth.append(
                    {
                        "boxes": target["boxes"].cpu(),
                        "labels": target["labels"].cpu(),
                    }
                )

    thresholds = [0.50 + 0.05 * index for index in range(10)]
    ap_table = np.zeros((len(thresholds), num_classes), dtype=np.float32)

    for class_id in range(num_classes):
        gt_per_image = [
            target["boxes"][target["labels"][:, class_id] > 0.5]
            for target in ground_truth
        ]
        total_gt = sum(len(boxes) for boxes in gt_per_image)
        if total_gt == 0:
            continue

        predictions = []
        for image_index, output in enumerate(detections):
            mask = output["labels"] == class_id
            for box, score in zip(
                output["boxes"][mask],
                output["scores"][mask],
            ):
                predictions.append((image_index, float(score), box))
        predictions.sort(key=lambda item: item[1], reverse=True)

        for threshold_index, iou_threshold in enumerate(thresholds):
            true_positive = np.zeros(len(predictions), dtype=np.float32)
            false_positive = np.zeros(len(predictions), dtype=np.float32)
            used = [np.zeros(len(boxes), dtype=bool) for boxes in gt_per_image]

            for index, (image_index, _, predicted_box) in enumerate(predictions):
                gt_boxes = gt_per_image[image_index]
                if len(gt_boxes) == 0:
                    false_positive[index] = 1
                    continue

                ious = box_iou(predicted_box.unsqueeze(0), gt_boxes)[0].numpy()
                best_index = int(ious.argmax())
                if (
                    ious[best_index] >= iou_threshold
                    and not used[image_index][best_index]
                ):
                    true_positive[index] = 1
                    used[image_index][best_index] = True
                else:
                    false_positive[index] = 1

            tp = np.cumsum(true_positive)
            fp = np.cumsum(false_positive)
            recalls = tp / max(total_gt, 1)
            precisions = tp / np.maximum(tp + fp, 1e-8)
            ap_table[threshold_index, class_id] = compute_ap(
                recalls,
                precisions,
            )

    return float(ap_table[0].mean()), float(ap_table.mean())
