# predict.py
import argparse
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from pid_train.lightning_modules.object_detection_lit_module import ObjectDetectionLitModule
from pid_train.datasets.coco_style_dataset import CocoStyleDataset
from utils.tile_utils import tile_image, merge_detections

# Pillow의 DecompressionBombWarning 비활성화
Image.MAX_IMAGE_PIXELS = None

def draw_predictions(image, predictions, class_map):
    """
    이미지에 예측 결과를 그립니다.
    """
    for box, score, label in zip(predictions['boxes'], predictions['scores'], predictions['labels']):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_map.get(label.item(), 'Unknown')
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 텍스트 그리기
        label_text = f"{class_name}: {score:.2f}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    return image

def main(args):
    print(f"Loading model from checkpoint: {args.checkpoint_path}")
    model = ObjectDetectionLitModule.load_from_checkpoint(args.checkpoint_path).to(args.device)
    model.eval()

    print(f"Loading image: {args.image_path}")
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {args.image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 클래스 이름 매핑을 위해 데이터셋 로드
    # annotation_file 경로는 실제 프로젝트에 맞게 수정해야 할 수 있습니다.
    print("Loading dataset to get class map...")
    dataset = CocoStyleDataset(image_dir="", annotation_file=args.annotation_path)
    # COCO의 cat_id -> 이름
    coco_id_to_name = {cat['id']: cat['name'] for cat in dataset.coco.dataset['categories']}
    # 연속적인 ID -> COCO cat_id -> 이름
    contiguous_id_to_name = {v: coco_id_to_name[k] for k, v in dataset.coco_cat_id_to_contiguous_id.items()}

    print("Tiling image...")
    tiles = tile_image(image, tile_size=args.tile_size, overlap=args.overlap)

    # 각 타일에 대한 예측 수행
    tile_results = []
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    print(f"Running inference on {len(tiles)} tiles...")
    with torch.no_grad():
        for tile, offset in tiles:
            transformed_tile = transform(image=tile)["image"].to(args.device)
            predictions = model([transformed_tile])
            
            # 결과를 CPU로 이동하고 필요한 정보만 추출
            pred = predictions[0]
            pred['tile_offset'] = offset
            tile_results.append(pred)

    print("Merging detections...")
    merged_predictions = merge_detections(
        tile_results, 
        original_shape=image.shape,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )

    # 최종 결과는 리스트 안에 딕셔너리 형태로 반환됨
    final_detections = merged_predictions[0] if merged_predictions else {'boxes': [], 'scores': [], 'labels': []}

    print(f"Found {len(final_detections['boxes'])} objects.")

    print("Drawing predictions on the original image...")
    output_image = draw_predictions(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), final_detections, contiguous_id_to_name)

    print(f"Saving output image to: {args.output_path}")
    cv2.imwrite(args.output_path, output_image)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiled inference for large images.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint file (.ckpt).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the large input image.")
    parser.add_argument("--output_path", type=str, default="output.png", help="Path to save the output image with predictions.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to the COCO annotation file to get class names.")
    parser.add_argument("--tile_size", type=int, default=640, help="Size of the tiles.")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap between tiles.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for Non-Maximum Suppression.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for filtering detections.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on (e.g., 'cuda', 'cpu').")
    
    args = parser.parse_args()
    main(args)
