from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou

from src.models import build_model
from utils.checkpoint import load_checkpoint


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_image(path: Path, image_size: int):
    with Image.open(path) as image:
        image.load()
        gray = np.asarray(image.convert("L"), dtype=np.uint8)

    original = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(gray, (image_size, image_size))
    tensor = torch.from_numpy(
        np.stack([resized, resized, resized], axis=-1).astype(np.float32) / 255.0
    ).permute(2, 0, 1).float()
    return tensor, original


def load_thresholds(
    path: Optional[str],
    class_names: List[str],
    default: float,
) -> List[float]:
    if not path:
        return [default] * len(class_names)

    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if "class_thresholds" in data:
        data = data["class_thresholds"]

    return [
        float(data.get(str(index), data.get(name, default)))
        for index, name in enumerate(class_names)
    ]


def cluster_detections(
    detection: Dict[str, torch.Tensor],
    thresholds: List[float],
    merge_iou: float,
    max_objects: int,
):
    candidates = []
    for box, score, label in zip(
        detection["boxes"].cpu(),
        detection["scores"].cpu(),
        detection["labels"].cpu(),
    ):
        class_id = int(label)
        score_value = float(score)
        if score_value >= thresholds[class_id]:
            candidates.append(
                {"box": box.float(), "class_id": class_id, "score": score_value}
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)

    clusters = []
    for candidate in candidates:
        selected = None
        best_iou = 0.0
        for cluster in clusters:
            iou = float(
                box_iou(
                    candidate["box"].unsqueeze(0),
                    cluster["box"].unsqueeze(0),
                )[0, 0]
            )
            if iou >= merge_iou and iou > best_iou:
                selected, best_iou = cluster, iou

        if selected is None:
            clusters.append(
                {
                    "box": candidate["box"].clone(),
                    "weighted_sum": candidate["box"] * candidate["score"],
                    "weight": candidate["score"],
                    "labels": {candidate["class_id"]: candidate["score"]},
                    "score": candidate["score"],
                }
            )
        else:
            selected["weighted_sum"] += candidate["box"] * candidate["score"]
            selected["weight"] += candidate["score"]
            selected["box"] = selected["weighted_sum"] / selected["weight"]
            class_id = candidate["class_id"]
            selected["labels"][class_id] = max(
                selected["labels"].get(class_id, 0.0),
                candidate["score"],
            )
            selected["score"] = max(selected["score"], candidate["score"])

    return sorted(
        clusters,
        key=lambda item: item["score"],
        reverse=True,
    )[:max_objects]


def draw_objects(
    image: np.ndarray,
    objects,
    class_names: List[str],
    image_size: int,
    legend_width: int,
):
    height, width = image.shape[:2]
    canvas = np.zeros((height, width + legend_width, 3), dtype=np.uint8)
    canvas[:, :width] = image
    canvas[:, width:] = 24

    scale_x, scale_y = width / image_size, height / image_size
    cv2.putText(
        canvas, "Multi-label detections", (width + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
    )

    legend_y = 75
    for index, obj in enumerate(objects, 1):
        box = obj["box"].numpy()
        x1, y1, x2, y2 = (
            int(box[0] * scale_x),
            int(box[1] * scale_y),
            int(box[2] * scale_x),
            int(box[3] * scale_y),
        )
        color = (255, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas, f"box{index}", (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

        labels = ", ".join(
            f"{class_names[class_id]} ({score:.3f})"
            for class_id, score in sorted(
                obj["labels"].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        cv2.putText(
            canvas, f"box{index}: {labels}", (width + 20, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (245, 245, 245), 1,
        )
        legend_y += 30
    return canvas


def run_inference(
    config,
    image_path: str,
    checkpoint_path: str,
    output_path: str,
    threshold_json: Optional[str] = None,
    show: bool = False,
):
    device = resolve_device(config.train.device)
    model = build_model(config, pretrained_backbone=False)
    load_checkpoint(model, checkpoint_path, device, strict=True)
    model.eval()

    # Inference candidate settings override training settings.
    model.score_threshold = float(config.inference.model_score_threshold)
    model.nms_threshold = float(config.inference.class_nms_threshold)
    model.detections_per_image = int(config.inference.detections_per_image)

    tensor, original = load_image(
        Path(image_path),
        int(config.data.image_size),
    )
    thresholds = load_thresholds(
        threshold_json,
        list(config.data.class_names),
        float(config.inference.default_class_threshold),
    )

    with torch.inference_mode():
        detection = model([tensor.to(device)])[0]

    objects = cluster_detections(
        detection,
        thresholds,
        float(config.inference.object_merge_iou),
        int(config.inference.max_objects),
    )
    result = draw_objects(
        original,
        objects,
        list(config.data.class_names),
        int(config.data.image_size),
        int(config.inference.legend_width),
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), result):
        raise RuntimeError(f"Failed to save: {destination}")
    print(f"[SAVE] {destination}")

    if show:
        cv2.imshow("MLOD inference", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
