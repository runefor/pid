import cv2
import numpy as np
from typing import List, Tuple, Dict, Generator
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

def generate_tiles(
    img_w: int, 
    img_h: int, 
    tile_size: int, 
    overlap: float
) -> Generator[Tuple[int, int, int, int], None, None]:
    """
    이미지 크기, 타일 크기, 오버랩 비율을 입력받아 
    각 타일의 좌표(x1, y1, x2, y2)를 생성하는 함수.

    Args:
        img_w (int): 이미지의 가로 크기 (width)
        img_h (int): 이미지의 세로 크기 (height)
        tile_size (int): 타일 한 변의 크기 (정방형 기준)
        overlap (float): 타일 간 중첩 비율 (0.0 ~ 1.0)

    Yields:
        Tuple[int, int, int, int]: 각 타일의 좌표 (x1, y1, x2, y2)
    """

    # stride 계산: 오버랩을 반영한 이동 간격
    stride: int = int(tile_size * (1 - overlap))
    
    # 이미지가 타일보다 작은 경우 보정
    tile_w: int = min(tile_size, img_w)
    tile_h: int = min(tile_size, img_h)
    
    # 세로 방향 타일 생성
    for y in range(0, img_h, stride):
        # 마지막 줄이 남을 경우 타일을 아래쪽에 맞춤
        if y + tile_h > img_h and y != 0:
            y = img_h - tile_h
        
        # 가로 방향 타일 생성
        for x in range(0, img_w, stride):
            # 마지막 열이 남을 경우 타일을 오른쪽에 맞춤
            if x + tile_w > img_w and x != 0:
                x = img_w - tile_w

            x1: int = x
            y1: int = y
            x2: int = min(x1 + tile_w, img_w)
            y2: int = min(y1 + tile_h, img_h)

            # 타일 좌표 반환
            yield (x1, y1, x2, y2)

        # 마지막 행까지 처리하면 루프 종료
        if y + tile_h >= img_h:
            break