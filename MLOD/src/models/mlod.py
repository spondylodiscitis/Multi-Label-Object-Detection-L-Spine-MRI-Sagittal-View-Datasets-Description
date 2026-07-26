from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision.models.convnext import (
    ConvNeXt_Large_Weights,
    convnext_large,
)
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import (
    AnchorGenerator,
    RegionProposalNetwork,
    RPNHead,
)
from torchvision.ops import MultiScaleRoIAlign, box_iou, nms

from src.losses import AsymmetricLoss
from src.models.layers import (
    MultiLabelBoxPredictor,
    TwoMLPHead,
    make_activation,
    make_norm_layer,
)
from utils.boxes import decode_boxes, encode_boxes


class MultiLabelCascadeRCNN(nn.Module):
    def __init__(self, config, pretrained_backbone: Optional[bool] = None) -> None:
        super().__init__()
        data_cfg = config.data
        model_cfg = config.model
        loss_cfg = config.loss

        self.num_classes = int(data_cfg.num_classes)
        self.image_size = int(data_cfg.image_size)
        self.num_stages = int(model_cfg.num_stages)

        positive_thresholds = list(model_cfg.stage_positive_iou_thresholds)
        negative_thresholds = list(model_cfg.stage_negative_iou_thresholds)
        while len(positive_thresholds) < self.num_stages:
            positive_thresholds.append(positive_thresholds[-1])
        while len(negative_thresholds) < self.num_stages:
            negative_thresholds.append(negative_thresholds[-1])
        self.positive_thresholds = positive_thresholds[: self.num_stages]
        self.negative_thresholds = negative_thresholds[: self.num_stages]

        if pretrained_backbone is None:
            pretrained_backbone = bool(model_cfg.backbone_pretrained)
        weights = (
            ConvNeXt_Large_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        convnext = convnext_large(weights=weights)
        self.backbone = convnext.features
        output_channels = 1536

        if model_cfg.use_neck:
            neck_channels = int(model_cfg.neck_channels)
            self.neck = nn.Sequential(
                nn.Conv2d(
                    output_channels,
                    neck_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                make_norm_layer(model_cfg.neck_norm, neck_channels),
                make_activation(model_cfg.neck_activation),
                nn.Conv2d(
                    neck_channels,
                    neck_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                make_norm_layer(model_cfg.neck_norm, neck_channels),
                make_activation(model_cfg.neck_activation),
            )
            output_channels = neck_channels
        else:
            self.neck = nn.Identity()

        if model_cfg.freeze_stem:
            for parameter in self.backbone[0].parameters():
                parameter.requires_grad = False

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),),
        )
        rpn_head = RPNHead(
            output_channels,
            anchor_generator.num_anchors_per_location()[0],
        )
        self.rpn = RegionProposalNetwork(
            anchor_generator,
            rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=int(model_cfg.roi_batch_size_per_image),
            positive_fraction=float(model_cfg.roi_positive_fraction),
            pre_nms_top_n={
                "training": int(model_cfg.rpn_pre_nms_top_n_train),
                "testing": int(model_cfg.rpn_pre_nms_top_n_test),
            },
            post_nms_top_n={
                "training": int(model_cfg.rpn_post_nms_top_n_train),
                "testing": int(model_cfg.rpn_post_nms_top_n_test),
            },
            nms_thresh=float(model_cfg.rpn_nms_threshold),
        )

        self.roi_pool = MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2,
        )
        representation_size = 1024
        self.box_head = TwoMLPHead(
            output_channels * 7 * 7,
            representation_size,
        )
        self.stage_predictors = nn.ModuleList(
            [
                MultiLabelBoxPredictor(
                    representation_size,
                    self.num_classes,
                    float(model_cfg.label_attention_tau),
                    bool(model_cfg.label_attention_learnable),
                )
                for _ in range(self.num_stages)
            ]
        )

        self.roi_batch_size = int(model_cfg.roi_batch_size_per_image)
        self.roi_positive_fraction = float(model_cfg.roi_positive_fraction)
        self.score_threshold = float(model_cfg.score_threshold)
        self.nms_threshold = float(model_cfg.nms_threshold)
        self.detections_per_image = int(model_cfg.detections_per_image)

        self.classification_loss = AsymmetricLoss(
            gamma_positive=float(loss_cfg.asl_gamma_positive),
            gamma_negative=float(loss_cfg.asl_gamma_negative),
            clip=float(loss_cfg.asl_clip),
            class_counts=list(data_cfg.class_counts),
            class_balance_beta=float(loss_cfg.class_balance_beta),
        )
        self.regression_loss = nn.SmoothL1Loss(
            beta=float(loss_cfg.regression_beta),
            reduction="mean",
        )

    def _sample_proposals(
        self,
        proposals: torch.Tensor,
        ground_truth_boxes: torch.Tensor,
        positive_iou_threshold: float,
        negative_iou_threshold: float,
    ):
        device = proposals.device
        if ground_truth_boxes.numel() == 0:
            count = min(self.roi_batch_size, len(proposals))
            negative = torch.randperm(len(proposals), device=device)[:count]
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, negative, empty

        ious = box_iou(ground_truth_boxes, proposals)
        max_ious, assignments = ious.max(dim=0)
        positive = torch.where(max_ious >= positive_iou_threshold)[0]
        negative = torch.where(max_ious < negative_iou_threshold)[0]

        positive_count = min(
            int(self.roi_batch_size * self.roi_positive_fraction),
            positive.numel(),
        )
        negative_count = min(
            self.roi_batch_size - positive_count,
            negative.numel(),
        )

        if positive_count:
            positive = positive[
                torch.randperm(positive.numel(), device=device)[:positive_count]
            ]
            matched = assignments[positive]
        else:
            positive = torch.empty(0, dtype=torch.long, device=device)
            matched = torch.empty(0, dtype=torch.long, device=device)

        if negative_count:
            negative = negative[
                torch.randperm(negative.numel(), device=device)[:negative_count]
            ]
        else:
            negative = torch.empty(0, dtype=torch.long, device=device)

        return positive, negative, matched

    def _stage_step(
        self,
        proposals: List[torch.Tensor],
        logits: torch.Tensor,
        deltas: torch.Tensor,
        image_sizes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]],
        positive_iou_threshold: float,
        negative_iou_threshold: float,
    ):
        refined_proposals = []
        classification_losses = []
        regression_losses = []
        offset = 0

        for image_index, image_proposals in enumerate(proposals):
            count = len(image_proposals)
            image_logits = logits[offset : offset + count]
            image_deltas = deltas[offset : offset + count]
            offset += count

            if targets is not None:
                target = targets[image_index]
                positive, negative, matched = self._sample_proposals(
                    image_proposals,
                    target["boxes"],
                    positive_iou_threshold,
                    negative_iou_threshold,
                )
                selected = torch.cat([positive, negative])

                if selected.numel():
                    classification_targets = torch.zeros(
                        (len(selected), self.num_classes),
                        dtype=torch.float32,
                        device=image_logits.device,
                    )
                    if positive.numel():
                        classification_targets[: len(positive)] = target["labels"][matched]
                    classification_losses.append(
                        self.classification_loss(
                            image_logits[selected],
                            classification_targets,
                        )
                    )
                else:
                    classification_losses.append(image_logits.sum() * 0.0)

                if positive.numel():
                    regression_losses.append(
                        self.regression_loss(
                            image_deltas[positive],
                            encode_boxes(
                                target["boxes"][matched],
                                image_proposals[positive],
                            ),
                        )
                    )
                else:
                    regression_losses.append(image_deltas.sum() * 0.0)

            refined = decode_boxes(image_deltas, image_proposals)
            height, width = image_sizes[image_index]
            refined[:, [0, 2]].clamp_(0, width)
            refined[:, [1, 3]].clamp_(0, height)
            valid = (
                (refined[:, 2] > refined[:, 0])
                & (refined[:, 3] > refined[:, 1])
            )
            refined_proposals.append(refined[valid])

        classification_loss = (
            torch.stack(classification_losses).mean()
            if classification_losses
            else None
        )
        regression_loss = (
            torch.stack(regression_losses).mean()
            if regression_losses
            else None
        )
        return classification_loss, regression_loss, refined_proposals

    def _inference_single(
        self,
        proposals: torch.Tensor,
        logits: torch.Tensor,
        deltas: torch.Tensor,
        image_size: Tuple[int, int],
    ):
        boxes = decode_boxes(deltas, proposals)
        height, width = image_size
        boxes[:, [0, 2]].clamp_(0, width)
        boxes[:, [1, 3]].clamp_(0, height)

        valid = (
            (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        boxes = boxes[valid]
        probabilities = torch.sigmoid(logits[valid])

        final_boxes, final_scores, final_labels = [], [], []
        for class_id in range(self.num_classes):
            scores = probabilities[:, class_id]
            keep = scores > self.score_threshold
            if not torch.any(keep):
                continue
            class_boxes = boxes[keep]
            class_scores = scores[keep]
            keep_indices = nms(
                class_boxes,
                class_scores,
                self.nms_threshold,
            )
            final_boxes.append(class_boxes[keep_indices])
            final_scores.append(class_scores[keep_indices])
            final_labels.append(
                torch.full(
                    (len(keep_indices),),
                    class_id,
                    dtype=torch.int64,
                    device=boxes.device,
                )
            )

        if not final_boxes:
            return {
                "boxes": boxes.new_zeros((0, 4)),
                "scores": boxes.new_zeros((0,)),
                "labels": torch.zeros(
                    (0,), dtype=torch.int64, device=boxes.device
                ),
            }

        boxes = torch.cat(final_boxes)
        scores = torch.cat(final_scores)
        labels = torch.cat(final_labels)
        if len(boxes) > self.detections_per_image:
            indices = scores.topk(self.detections_per_image).indices
            boxes, scores, labels = (
                boxes[indices],
                scores[indices],
                labels[indices],
            )
        return {"boxes": boxes, "scores": scores, "labels": labels}

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        if self.training and targets is None:
            raise ValueError("Targets are required in training mode.")

        image_sizes = [image.shape[-2:] for image in images]
        image_list = ImageList(torch.stack(images), image_sizes)

        features = self.neck(self.backbone(image_list.tensors))
        feature_map = {"0": features}
        rpn_targets = (
            [{"boxes": target["boxes"]} for target in targets]
            if self.training
            else None
        )
        proposals, rpn_losses = self.rpn(
            image_list,
            feature_map,
            rpn_targets,
        )

        stage_losses = {}
        current_proposals = proposals
        last_logits = last_deltas = last_input_proposals = None

        for stage_index in range(self.num_stages):
            pooled = self.roi_pool(
                feature_map,
                current_proposals,
                image_list.image_sizes,
            )
            representation = self.box_head(pooled)
            logits, deltas = self.stage_predictors[stage_index](representation)

            cls_loss, reg_loss, refined = self._stage_step(
                current_proposals,
                logits,
                deltas,
                image_list.image_sizes,
                targets,
                self.positive_thresholds[stage_index],
                self.negative_thresholds[stage_index],
            )
            if targets is not None:
                stage_losses[f"loss_cls_stage{stage_index + 1}"] = cls_loss
                stage_losses[f"loss_box_stage{stage_index + 1}"] = reg_loss

            last_logits = logits
            last_deltas = deltas
            last_input_proposals = current_proposals
            current_proposals = refined

        if targets is not None:
            return {**rpn_losses, **stage_losses}

        detections = []
        offset = 0
        for image_index, proposals_for_image in enumerate(last_input_proposals):
            count = len(proposals_for_image)
            detections.append(
                self._inference_single(
                    proposals_for_image,
                    last_logits[offset : offset + count],
                    last_deltas[offset : offset + count],
                    image_list.image_sizes[image_index],
                )
            )
            offset += count
        return detections


def build_model(config, pretrained_backbone: Optional[bool] = None):
    return MultiLabelCascadeRCNN(config, pretrained_backbone)
