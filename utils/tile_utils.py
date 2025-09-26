import cv2
import numpy as np
from typing import List, Tuple, Dict
import torch

def tile_image(image: np.ndarray, tile_size: int = 640, overlap: float = 0.2) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    이미지를 타일로 나눔. 오버랩 포함.
    Returns: (타일 이미지, (타일 x 오프셋, 타일 y 오프셋)) 리스트
    """
    h, w = image.shape[:2]
    stride = int(tile_size * (1 - overlap))
    tiles = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 타일 추출 (패딩으로 크기 맞춤)
            tile = image[y:y+tile_size, x:x+tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            tiles.append((tile, (x, y)))
    return tiles

def merge_detections(tiles_results: List[Dict], original_shape: Tuple[int, int], 
                     iou_threshold: float = 0.5, conf_threshold: float = 0.5) -> List[Dict]:
    """
    타일별 detections 병합 + NMS.
    tiles_results: [{'boxes': torch.Tensor [N,4], 'scores': torch.Tensor [N], 'labels': torch.Tensor [N], 'tile_offset': (x,y)}]
    Returns: 병합된 detections (원본 좌표 기준)
    """
    all_boxes = []
    all_scores = []
    all_labels = []
    
    for res in tiles_results:
        offset_x, offset_y = res['tile_offset']
        boxes = res['boxes'].cpu().numpy()  # [N, 4] (x1,y1,x2,y2 normalized? -> absolute로 가정)
        scores = res['scores'].cpu().numpy()
        labels = res['labels'].cpu().numpy()
        
        # 타일 오프셋 적용 (정규화되지 않은 absolute 좌표 가정)
        boxes[:, [0, 2]] += offset_x  # x1, x2
        boxes[:, [1, 3]] += offset_y  # y1, y2
        
        # conf 필터링
        mask = scores > conf_threshold
        all_boxes.extend(boxes[mask])
        all_scores.extend(scores[mask])
        all_labels.extend(labels[mask])
    
    if not all_boxes:
        return []
    
    # torch로 변환 후 NMS
    all_boxes = torch.tensor(all_boxes, dtype=torch.float32)
    all_scores = torch.tensor(all_scores, dtype=torch.float32)
    all_labels = torch.tensor(all_labels, dtype=torch.long)
    
    # torchvision.ops.nms 사용 (bbox format: xyxy)
    from torchvision.ops import nms
    keep = nms(all_boxes, all_scores, iou_threshold)
    
    return [{'boxes': all_boxes[keep], 'scores': all_scores[keep], 'labels': all_labels[keep]}]