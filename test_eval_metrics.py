import json
import os
import torch

from pid_train.metrics.eval_metrics import ObjectDetectionMetrics, ConfusionMatrix, ObjectDetectionAP


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