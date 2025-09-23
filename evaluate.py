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
import json
import tempfile
import os
from typing import List, Dict, Any

import torch
from torchmetrics import Metric
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
        pred_labels = pred['labels']  # [N], 클래스 0~145 (또는 1~146; 조정 필요)
        gt_boxes = target['boxes']
        gt_labels = target['labels']  # [M]
        
        # 클래스 루프
        for cls in range(self.num_classes):
            mask_pred = (pred_labels == cls)
            mask_gt = (gt_labels == cls)
            pred_cls_boxes = pred_boxes[mask_pred]
            gt_cls_boxes = gt_boxes[mask_gt]
            
            if len(pred_cls_boxes) == 0 and len(gt_cls_boxes) == 0:
                continue
            if len(pred_cls_boxes) == 0:
                self.false_negatives[cls] += len(gt_cls_boxes)
                continue
            if len(gt_cls_boxes) == 0:
                self.false_positives[cls] += len(pred_cls_boxes)
                continue
            
            # Greedy IoU 매칭
            iou_matrix = box_iou(pred_cls_boxes, gt_cls_boxes)
            tp = 0
            matched_preds = set()
            for i in range(len(gt_cls_boxes)):
                best_j = -1
                best_iou = 0.0
                for j in range(len(pred_cls_boxes)):
                    if j in matched_preds:
                        continue
                    if iou_matrix[j, i] > best_iou:
                        best_iou = iou_matrix[j, i]
                        best_j = j
                if best_iou > self.iou_threshold:
                    tp += 1
                    matched_preds.add(best_j)
            
            fp = len(pred_cls_boxes) - len(matched_preds)
            fn = len(gt_cls_boxes) - tp
            self.true_positives[cls] += tp
            self.false_positives[cls] += fp
            self.false_negatives[cls] += fn
            

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
                

class ObjectDetectionAP(Metric):
    """
    TorchMetrics 스타일의 mAP와 클래스별 AP 계산 클래스.
    pycocotools를 사용해 COCO 형식으로 평가합니다.
    - update(): 예측 결과를 COCO 형식 리스트로 누적 (이미지 ID, category_id, bbox, score 포함).
    - compute(): 임시 JSON 파일 생성 후 COCOeval로 mAP@0.5:0.95, mAP@0.5, 클래스별 AP 반환.
    
    사용법:
    - GT는 COCO 형식 annotations.json 파일 경로를 __init__에 전달.
    - preds는 [{'image_id': int, 'category_id': int, 'bbox': [x,y,w,h], 'score': float}, ...] 형태.
    - 클래스 ID: 1~num_classes (배경 0 제외).
    """
    def __init__(
        self,
        annotations_file: str,  # GT annotations.json 경로
        num_classes: int = 146,
        iou_type: str = 'bbox',  # 'bbox' or 'segm'
        max_dets: int = 100,
        dist_sync_on_step: bool = False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.annotations_file = annotations_file
        self.num_classes = num_classes
        self.iou_type = iou_type
        self.max_dets = max_dets
        
        # 상태: 예측 결과 리스트 누적
        self.add_state("predictions", default=[], dist_reduce_fx="cat")  # 리스트지만 TorchMetrics에서 텐서처럼 취급; 실제론 리스트

        # COCO GT 로드 (초기화 시)
        self.coco_gt = COCO(annotations_file)
        self.category_ids = self.coco_gt.getCatIds()  # 클래스 ID 리스트
        if len(self.category_ids) != num_classes:
            raise ValueError(f"GT 파일의 클래스 수({len(self.category_ids)})가 num_classes({num_classes})와 맞지 않습니다.")

    def update(self, preds: List[Dict[str, Any]]):
        """
        예측 결과를 누적. preds: COCO 형식 리스트.
        예: [{'image_id': 1, 'category_id': 1, 'bbox': [10,10,20,20], 'score': 0.9}, ...]
        """
        self.predictions.extend(preds)

    def compute(self) -> Dict[str, Any]:
        if not self.predictions:
            return {"map": 0.0, "map_50": 0.0, "per_class_ap": [0.0] * self.num_classes}

        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.predictions, f)
            pred_file = f.name

        # COCO DT 로드
        coco_dt = self.coco_gt.loadRes(pred_file)
        
        # 평가 실행
        coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)
        coco_eval.params.maxDets = [0, self.max_dets, 1000]  # max detections 제한
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # 결과 추출: mAP@0.5:0.95 (stats[0]), mAP@0.5 (stats[1])
        stats = coco_eval.stats
        map_score = stats[0]  # mAP@0.5:0.95
        map_50 = stats[1]    # mAP@0.5
        
        # 클래스별 AP: coco_eval.eval['precision']에서 추출 (11-point 평균, but all-point approx)
        per_class_ap = [0.0] * self.num_classes
        for i, cat_id in enumerate(self.category_ids):
            # coco_eval.eval['precision'][0, :, i, 0, -1] : IoU=0.5:0.95, area=all, maxDets=100의 AP
            ap = coco_eval.eval['precision'][0, :, i, 0, -1].mean() if len(coco_eval.eval['precision']) > 0 else 0.0
            per_class_ap[cat_id - 1] = ap  # 클래스 ID 1~146 → 인덱스 0~145

        # 임시 파일 삭제
        os.unlink(pred_file)

        # 예측 리스트 리셋 (다음 에포크용)
        self.predictions = []

        return {
            "map": float(map_score),
            "map_50": float(map_50),
            "per_class_ap": per_class_ap  # [146개] 리스트
        }

    def reset(self):
        """상태 리셋"""
        super().reset()
        self.predictions = []



# 테스트 코드 (COCO 형식 GT와 preds로 모든 메트릭스 계산)
if __name__ == "__main__":
    # 더미 GT JSON 생성 (COCO 형식: 1 이미지, 클래스 1 어노테이션, bbox=[x,y,w,h])
    dummy_gt = {
        "info": {
            "description": "Dummy Dataset for Testing",
            "version": "1.0",
            "year": 2025,
            "contributor": "Test User",
            "date_created": "2025/09/23"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Dummy License",
                "url": "http://example.com/dummy-license"
            }
        ],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, 147)],  # 1~146 클래스
        "images": [{"id": 1, "width": 640, "height": 480}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 10.0, 10.0], "area": 100.0, "iscrowd": 0}
        ]
    }
    gt_file = "dummy_gt.json"
    with open(gt_file, 'w') as f:
        json.dump(dummy_gt, f)
    
    # 더미 예측 (COCO 형식: bbox=[x,y,w,h], score 포함)
    dummy_preds_coco = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 10.0, 10.0], "score": 0.99}
    ]
    
    # F1과 CM을 위한 boxes/labels 형식 변환 (COCO bbox [x,y,w,h] → [x1,y1,x2,y2])
    x, y, w, h = 10.0, 10.0, 10.0, 10.0
    dummy_boxes_pred = torch.tensor([[x, y, x + w, y + h]])
    dummy_labels_pred = torch.tensor([1])
    dummy_boxes_target = torch.tensor([[x, y, x + w, y + h]])
    dummy_labels_target = torch.tensor([1])
    preds_boxes = [{'boxes': dummy_boxes_pred, 'labels': dummy_labels_pred}]
    targets_boxes = [{'boxes': dummy_boxes_target, 'labels': dummy_labels_target}]
    
    # 인스턴스 생성
    f1_metric = ObjectDetectionMetrics(num_classes=146)
    cm_metric = ConfusionMatrix(num_classes=146)
    ap_metric = ObjectDetectionAP(annotations_file=gt_file, num_classes=146)
    
    # 업데이트
    f1_metric.update(preds_boxes, targets_boxes)
    cm_metric.update(preds_boxes, targets_boxes)
    ap_metric.update(dummy_preds_coco)
    
    # 결과
    f1_results = f1_metric.compute()
    cm = cm_metric.compute()
    ap_results = ap_metric.compute()
    
    print(f1_results)  # {'macro_f1_score': ~0.0069, ...} (1/146 클래스만)
    print(cm.shape)  # torch.Size([146, 146])
    print(ap_results)  # {'map': ~1.0, 'map_50': ~1.0, 'per_class_ap': [1.0, 0.0, ..., 0.0]}
    
    # 파일 정리
    os.unlink(gt_file)