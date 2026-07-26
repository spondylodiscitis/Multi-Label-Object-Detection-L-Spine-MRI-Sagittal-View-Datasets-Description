from __future__ import annotations

import albumentations as A
import cv2


def build_train_transform():
    """BBox-aware augmentation preserving multi-label row order."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.03,
                scale_limit=0.10,
                rotate_limit=10,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=0.0,
            min_visibility=0.0,
            label_fields=["bbox_indices"],
        ),
    )
