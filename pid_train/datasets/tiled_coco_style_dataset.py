# pid_train/datasets/tiled_coco_style_dataset.py
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np

class TiledCocoStyleDataset(Dataset):
    """
    Tiled-Training을 위한 COCO 형식 데이터셋 클래스.
    - __init__에서 모든 타일 정보를 미리 생성합니다.
    - __getitem__에서는 해당 인덱스의 타일과 어노테이션을 반환합니다.
    """
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transforms: Optional[Callable] = None,
        tile_size: int = 640,
        overlap: float = 0.2,
    ):
        self.image_dir = image_dir
        self.transforms = transforms
        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap))

        print("[Tiled Dataset] Loading annotations...")
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        # COCO category_id를 [1, num_categories] 범위로 리매핑
        coco_cat_ids = sorted(self.coco.getCatIds())
        self.coco_cat_id_to_contiguous_id = {
            coco_id: i + 1 for i, coco_id in enumerate(coco_cat_ids)
        }
        self.num_classes = len(coco_cat_ids)
        print(f"[Tiled Dataset] 총 {self.num_classes}개의 클래스를 발견했습니다.")

        print("[Tiled Dataset] Pre-calculating all possible tiles...")
        self.tiles = self._create_tiles()
        print(f"[Tiled Dataset] Created {len(self.tiles)} tiles in total.")

    def _create_tiles(self):
        tiles = []
        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_h, img_w = img_info['height'], img_info['width']

            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            annotations = self.coco.loadAnns(ann_ids)

            for y in range(0, img_h, self.stride):
                for x in range(0, img_w, self.stride):
                    tile_x1, tile_y1 = x, y
                    tile_x2, tile_y2 = x + self.tile_size, y + self.tile_size

                    tile_annotations = []
                    for ann in annotations:
                        ann_x1, ann_y1, ann_w, ann_h = ann['bbox']
                        ann_x2, ann_y2 = ann_x1 + ann_w, ann_y1 + ann_h

                        # 타일과 어노테이션이 겹치는지 확인
                        inter_x1 = max(tile_x1, ann_x1)
                        inter_y1 = max(tile_y1, ann_y1)
                        inter_x2 = min(tile_x2, ann_x2)
                        inter_y2 = min(tile_y2, ann_y2)

                        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                            # 타일 좌표 기준으로 bbox 변환
                            new_bbox = [
                                inter_x1 - tile_x1,
                                inter_y1 - tile_y1,
                                inter_x2 - tile_x1,
                                inter_y2 - tile_y1,
                            ]
                            tile_annotations.append({
                                'bbox': new_bbox,
                                'category_id': self.coco_cat_id_to_contiguous_id[ann["category_id"]]
                            })
                    
                    # 타일 내에 어노테이션이 하나라도 있는 경우에만 학습 데이터로 추가
                    if tile_annotations:
                        tiles.append({
                            'image_id': img_id,
                            'image_path': os.path.join(self.image_dir, img_info['file_name']),
                            'tile_coords': (x, y, self.tile_size, self.tile_size),
                            'annotations': tile_annotations
                        })
        return tiles

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        tile_info = self.tiles[index]
        
        img = Image.open(tile_info['image_path']).convert("RGB")
        
        # 타일 crop
        x, y, w, h = tile_info['tile_coords']
        tile_img = img.crop((x, y, x + w, y + h))
        tile_img_np = np.array(tile_img)

        # 어노테이션 준비
        boxes = [ann['bbox'] for ann in tile_info['annotations']]
        labels = [ann['category_id'] for ann in tile_info['annotations']]

        if self.transforms:
            transformed = self.transforms(
                image=tile_img_np,
                bboxes=boxes,
                labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

        return image, target
