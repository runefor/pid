
import argparse
import json
import os
from typing import List, Dict, Any

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
import numpy as np

from pid_train.lightning_modules.object_detection_lit_module import ObjectDetectionLitModule
from pid_train.datasets.coco_style_dataset import CocoStyleDataset
from pid_train.metrics.eval_metrics import ObjectDetectionAP, ObjectDetectionMetrics, ConfusionMatrix

# COCO 카테고리 ID를 연속적인 ID로 매핑하는 유틸리티
def get_coco_cat_id_to_contiguous_id(coco_api):
    coco_cat_ids = sorted(coco_api.getCatIds())
    return {coco_id: i + 1 for i, coco_id in enumerate(coco_cat_ids)}

def slide_window_predict(
    model: ObjectDetectionLitModule,
    image_path: str,
    tile_size: int,
    overlap: float,
    device: torch.device,
    transforms: Any,
):
    """슬라이딩 윈도우를 사용하여 전체 이미지에 대한 예측을 수행합니다."""
    stride = int(tile_size * (1 - overlap))
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    
    all_preds = []

    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            tile_x1, tile_y1 = x, y
            tile_x2, tile_y2 = min(x + tile_size, img_w), min(y + tile_size, img_h)
            
            tile_img = img.crop((tile_x1, tile_y1, tile_x2, tile_y2))
            
            # 모델 입력 형식에 맞게 변환
            tile_tensor = transforms(tile_img).unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model(tile_tensor)

            # 예측 결과를 전체 이미지 좌표로 변환
            for pred in preds:
                boxes = pred["boxes"]
                labels = pred["labels"]
                scores = pred["scores"]

                # 타일 좌표를 전체 이미지 좌표로 변환
                boxes[:, 0] += tile_x1
                boxes[:, 1] += tile_y1
                boxes[:, 2] += tile_x1
                boxes[:, 3] += tile_y1

                all_preds.append({"boxes": boxes, "labels": labels, "scores": scores})

    # 모든 타일의 예측을 하나로 모음
    if not all_preds:
        return {"boxes": torch.empty(0, 4), "labels": torch.empty(0), "scores": torch.empty(0)}

    final_boxes = torch.cat([p["boxes"] for p in all_preds], dim=0)
    final_labels = torch.cat([p["labels"] for p in all_preds], dim=0)
    final_scores = torch.cat([p["scores"] for p in all_preds], dim=0)

    # NMS를 적용하여 중복된 박스 제거
    keep_indices = torchvision.ops.nms(final_boxes, final_scores, iou_threshold=0.5)
    
    final_boxes = final_boxes[keep_indices]
    final_labels = final_labels[keep_indices]
    final_scores = final_scores[keep_indices]

    return {"boxes": final_boxes, "labels": final_labels, "scores": final_scores}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 로드
    model = ObjectDetectionLitModule.load_from_checkpoint(args.checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # 데이터셋 및 데이터로더 준비 (평가는 전체 이미지를 대상으로 함)
    # 간단한 ToTensor 변환만 적용
    val_transforms = lambda img: F.to_tensor(img)
    dataset = CocoStyleDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        transforms=None, # 예측 시에는 이미지 단위로 직접 변환
    )
    
    coco_cat_id_to_contiguous_id = get_coco_cat_id_to_contiguous_id(dataset.coco)
    contiguous_id_to_coco_cat_id = {v: k for k, v in coco_cat_id_to_contiguous_id.items()}

    # 평가 메트릭 초기화
    ap_metric = ObjectDetectionAP(annotations_file=args.annotation_file, num_classes=model.model.num_classes)
    f1_metric = ObjectDetectionMetrics(num_classes=model.model.num_classes)
    cm_metric = ConfusionMatrix(num_classes=model.model.num_classes)
    
    coco_preds = []
    
    print("Starting evaluation...")
    for i in tqdm(range(len(dataset))):
        # Ground Truth 로드
        _, target, image_id = dataset[i]
        image_path = dataset.get_image_info(i)[1]

        # 슬라이딩 윈도우로 예측 수행
        preds = slide_window_predict(
            model=model,
            image_path=image_path,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=device,
            transforms=val_transforms,
        )

        # F1, CM 계산을 위한 데이터 준비
        # Target의 bbox를 (x,y,w,h) -> (x1,y1,x2,y2)로 변환
        gt_boxes_xywh = target['boxes']
        gt_boxes_xyxy = gt_boxes_xywh.clone()
        if gt_boxes_xyxy.shape[0] > 0:
            gt_boxes_xyxy[:, 2] = gt_boxes_xywh[:, 0] + gt_boxes_xywh[:, 2]
            gt_boxes_xyxy[:, 3] = gt_boxes_xywh[:, 1] + gt_boxes_xywh[:, 3]
        
        target_for_metric = [{'boxes': gt_boxes_xyxy.to(device), 'labels': target['labels'].to(device)}]
        preds_for_metric = [preds]

        # F1, CM 메트릭 업데이트
        f1_metric.update(preds_for_metric, target_for_metric)
        cm_metric.update(preds_for_metric, target_for_metric)

        # AP 계산을 위한 COCO 형식 변환
        for j in range(len(preds["boxes"])):
            box = preds["boxes"][j].cpu().numpy()
            label = preds["labels"][j].cpu().item()
            score = preds["scores"][j].cpu().item()
            
            coco_cat_id = contiguous_id_to_coco_cat_id.get(label)
            if coco_cat_id is None:
                continue

            x1, y1, x2, y2 = box
            bbox = [x1, y1, x2 - x1, y2 - y1]

            coco_preds.append({
                "image_id": image_id,
                "category_id": coco_cat_id,
                "bbox": bbox,
                "score": score,
            })

    # AP 메트릭 계산 및 출력
    print("\nCalculating AP metrics (COCO standard)...")
    ap_metric.update(coco_preds)
    ap_results = ap_metric.compute()
    print(json.dumps(ap_results, indent=4))

    # F1/Precision/Recall 메트릭 계산 및 출력
    print("\nCalculating F1/Precision/Recall...")
    f1_results = f1_metric.compute()
    print(json.dumps(f1_results, indent=4))

    # Confusion Matrix 계산 및 저장
    print("\nCalculating Confusion Matrix...")
    cm_tensor = cm_metric.compute()
    cm_output_path = "confusion_matrix.pt"
    torch.save(cm_tensor, cm_output_path)
    print(f"Confusion Matrix Shape: {cm_tensor.shape}")
    print(f"Confusion Matrix saved to {cm_output_path}")

    # 예측 결과를 파일로 저장 (옵션)
    if args.output_path:
        with open(args.output_path, 'w') as f:
            json.dump(coco_preds, f, indent=4)
        print(f"\nPredictions saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Tiled Object Detection Model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing validation images.")
    parser.add_argument("--annotation_file", type=str, required=True, help="Path to the COCO annotation file for validation.")
    parser.add_argument("--tile_size", type=int, default=640, help="Tile size used during training.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap percentage between tiles.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional path to save the COCO-formatted prediction JSON file.")
    
    args = parser.parse_args()
    main(args)
