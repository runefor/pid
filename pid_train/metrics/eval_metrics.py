import json
import tempfile
import os
from typing import List, Dict, Any

import torch
from torchmetrics import Metric
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

"""
IoU
Recall
Precision
f1-score -> 지금 클래스 불균형이 심하니까 이거 쓰는게 Recall 쓰는 것 보다 더 좋을듯
mAP -> pycocotools 이걸로 구현하는 것이 좋다고 함!!
클래스별 AP -> 이것도 마찬가지!!!
Confusion Matrix

위와 같은 평가 지표를 계산하는 함수 혹은 클래스가 필요함.
"""

class ObjectDetectionMetrics(Metric): # TODO: metric.py로 옮기기
    def __init__(self, num_classes: int = 146, iou_threshold: float = 0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        # 스칼라 → 벡터로 변경: 클래스별 누적
        self.add_state("true_positives", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.zeros(num_classes, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: list, targets: list):
        for pred, target in zip(preds, targets):
            self._calculate_metrics(pred, target)

    def compute(self):
        # 클래스별 Precision/Recall/F1 계산 후 macro 평균
        precisions = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recalls = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        macro_precision = precisions.mean().item()
        macro_recall = recalls.mean().item()
        macro_f1 = f1_scores.mean().item()
        
        return {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1_score": macro_f1,
            # 옵션: 클래스별 F1 (리스트로 반환)
            "per_class_f1": f1_scores.tolist()
        }

    def _calculate_metrics(self, pred: dict, target: dict):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        if len(pred_boxes) == 0:
            if len(gt_boxes) > 0:
                for label in gt_labels:
                    if label < self.num_classes:
                        self.false_negatives[label] += 1
            return

        if len(gt_boxes) == 0:
            for label in pred_labels:
                if label < self.num_classes:
                    self.false_positives[label] += 1
            return

        # Sort predictions by score
        if 'scores' in pred:
            scores = pred['scores']
            sorted_indices = torch.argsort(scores, descending=True)
            pred_boxes = pred_boxes[sorted_indices]
            pred_labels = pred_labels[sorted_indices]

        iou_matrix = box_iou(pred_boxes, gt_boxes)

        # Find the best match for each ground truth box
        gt_matches = torch.zeros(len(gt_boxes), dtype=torch.bool, device=pred_boxes.device)
        pred_matches = torch.zeros(len(pred_boxes), dtype=torch.bool, device=pred_boxes.device)

        for i in range(len(gt_boxes)):
            # Find the best prediction for this ground truth box
            best_match_iou = -1
            best_match_idx = -1

            for j in range(len(pred_boxes)):
                if pred_labels[j] < self.num_classes and gt_labels[i] < self.num_classes and pred_labels[j] == gt_labels[i] and not pred_matches[j]:
                    iou = iou_matrix[j, i]
                    if iou > best_match_iou:
                        best_match_iou = iou
                        best_match_idx = j

            if best_match_iou > self.iou_threshold:
                if not gt_matches[i] and not pred_matches[best_match_idx]:
                    if gt_labels[i] < self.num_classes:
                        self.true_positives[gt_labels[i]] += 1
                    gt_matches[i] = True
                    pred_matches[best_match_idx] = True

        # Process unmatched predictions as false positives
        for j in range(len(pred_boxes)):
            if not pred_matches[j]:
                if pred_labels[j] < self.num_classes:
                    self.false_positives[pred_labels[j]] += 1

        # Process unmatched ground truths as false negatives
        for i in range(len(gt_boxes)):
            if not gt_matches[i]:
                if gt_labels[i] < self.num_classes:
                    self.false_negatives[gt_labels[i]] += 1
            

class ConfusionMatrix(Metric): # TODO: metric.py로 옮기기
    def __init__(self, num_classes: int = 146, iou_threshold: float = 0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        # Confusion Matrix: [num_classes, num_classes]
        self.add_state("cm", default=torch.zeros(num_classes, num_classes, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: list, targets: list):
        for pred, target in zip(preds, targets):
            self._update_cm(pred, target)

    def compute(self):
        return self.cm  # [146, 146] 텐서 반환 (시각화용: matplotlib으로 heatmap)

    def _update_cm(self, pred: dict, target: dict):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        
        # 클래스별 루프
        for cls_pred in range(self.num_classes):
            mask_pred = (pred_labels == cls_pred)
            pred_cls_boxes = pred_boxes[mask_pred]
            num_pred = len(pred_cls_boxes)
            
            for cls_gt in range(self.num_classes):
                mask_gt = (gt_labels == cls_gt)
                gt_cls_boxes = gt_boxes[mask_gt]
                num_gt = len(gt_cls_boxes)
                
                if num_pred == 0 and num_gt == 0:
                    continue
                if num_pred == 0:
                    self.cm[cls_pred, cls_gt] += num_gt  # FN
                    continue
                if num_gt == 0:
                    self.cm[cls_pred, 0] += num_pred  # FP (배경으로)
                    continue
                
                # IoU 매칭으로 TP/FP/FN 결정
                iou_matrix = box_iou(pred_cls_boxes, gt_cls_boxes)
                matched = torch.zeros(num_pred, dtype=torch.bool)
                tp_count = 0
                for i in range(num_gt):
                    best_idx = iou_matrix[:, i].argmax()
                    if iou_matrix[best_idx, i] > self.iou_threshold and not matched[best_idx]:
                        self.cm[cls_pred, cls_gt] += 1  # TP
                        matched[best_idx] = True
                        tp_count += 1
                    else:
                        self.cm[cls_pred, cls_gt] += 1  # FN (per GT)
                
                # 남은 pred는 FP (배경)
                fp_count = num_pred - matched.sum().item()
                self.cm[cls_pred, 0] += fp_count
                

from torchmetrics.detection import MeanAveragePrecision


class ObjectDetectionAP(MeanAveragePrecision):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)