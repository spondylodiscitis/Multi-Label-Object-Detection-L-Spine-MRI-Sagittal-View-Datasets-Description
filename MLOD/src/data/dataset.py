from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from utils.boxes import sanitize_boxes_and_labels

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SpineDataset(Dataset):
    """YOLO-style multi-label spine MRI dataset."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        root: str | Path,
        split: str,
        num_classes: int,
        image_size: int = 640,
        transform=None,
        min_box_wh: float = 2.0,
    ) -> None:
        self.root = Path(root).expanduser()
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.transform = transform
        self.min_box_wh = float(min_box_wh)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.rglob("*")):
            if image_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                continue
            label_path = (
                self.label_dir / image_path.relative_to(self.image_dir)
            ).with_suffix(".txt")
            if label_path.exists():
                self.samples.append((image_path, label_path))

        if not self.samples:
            raise RuntimeError(f"No paired samples found under: {self.image_dir}")

        print(f"[DATA] split={split} samples={len(self.samples):,}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_labels(self, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        boxes = []
        labels = []

        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"Malformed annotation: {label_path}:{line_number}: {line}"
                )

            try:
                class_ids = sorted(
                    set(int(token) for token in parts[0].split(",") if token.strip())
                )
                center_x, center_y, width, height = map(float, parts[1:5])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid annotation: {label_path}:{line_number}: {line}"
                ) from exc

            multi_hot = np.zeros(self.num_classes, dtype=np.float32)
            for class_id in class_ids:
                if not 0 <= class_id < self.num_classes:
                    raise ValueError(
                        f"Class ID {class_id} outside [0, {self.num_classes - 1}] "
                        f"at {label_path}:{line_number}"
                    )
                multi_hot[class_id] = 1.0

            center_x *= self.image_size
            center_y *= self.image_size
            width *= self.image_size
            height *= self.image_size

            boxes.append(
                [
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ]
            )
            labels.append(multi_hot)

        if not boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, self.num_classes), dtype=np.float32),
            )

        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
        )

    def __getitem__(self, index: int):
        image_path, label_path = self.samples[index]

        with Image.open(image_path) as image:
            image.load()
            image = image.convert("L").resize(
                (self.image_size, self.image_size),
                Image.Resampling.BILINEAR,
            )
            gray = np.asarray(image, dtype=np.uint8)

        image_array = np.stack([gray, gray, gray], axis=-1)
        boxes, labels = self._read_labels(label_path)

        boxes, labels = sanitize_boxes_and_labels(
            boxes,
            labels,
            image_size=self.image_size,
            min_wh=self.min_box_wh,
        )

        if self.transform is not None and len(boxes):
            augmented = self.transform(
                image=image_array,
                bboxes=boxes.tolist(),
                bbox_indices=list(range(len(boxes))),
            )
            image_array = augmented["image"]
            retained_indices = np.asarray(
                augmented["bbox_indices"], dtype=np.int64
            )
            boxes = np.asarray(augmented["bboxes"], dtype=np.float32).reshape(-1, 4)
            labels = labels[retained_indices] if len(retained_indices) else labels[:0]

            boxes, labels = sanitize_boxes_and_labels(
                boxes,
                labels,
                image_size=self.image_size,
                min_wh=self.min_box_wh,
            )

        image_tensor = torch.from_numpy(
            image_array.astype(np.float32) / 255.0
        ).permute(2, 0, 1).contiguous()

        target: Dict[str, torch.Tensor] = {
            "boxes": torch.from_numpy(boxes).float(),
            "labels": torch.from_numpy(labels).float(),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
