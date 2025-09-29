import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import numpy as np

class CocoStyleDataset(Dataset):
    """
    COCO 형식의 데이터셋을 위한 PyTorch Dataset 클래스.
    COCO의 category_id를 [1, num_categories] 범위로 리매핑합니다.
    """
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transforms: Optional[Callable] = None,
    ):
        self.image_dir = image_dir
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))

        coco_cat_ids = sorted(self.coco.getCatIds())
        self.coco_cat_id_to_contiguous_id = {
            coco_id: i + 1 for i, coco_id in enumerate(coco_cat_ids)
        }
        self.num_classes = len(coco_cat_ids)
        print(f"[Dataset] 총 {self.num_classes}개의 클래스를 발견했습니다.")
        print(f"[Dataset] Category IDs remapped to 1-{self.num_classes}")

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_anns = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]["file_name"]
        img_path = os.path.join(self.image_dir, path)

        img = Image.open(img_path).convert("RGB")

        boxes = [ann["bbox"] for ann in coco_anns]
        labels = [self.coco_cat_id_to_contiguous_id[ann["category_id"]] for ann in coco_anns]

        # 데이터 유효성 검사
        valid_boxes = []
        valid_labels = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            if w > 0 and h > 0:
                valid_boxes.append(box)
                valid_labels.append(labels[i])
            else:
                print(f"[WARNING] Image {img_id}: Invalid box [x={x}, y={y}, w={w}, h={h}] found. Skipping annotation.")

        for label in valid_labels:
            if not (1 <= label <= 145):
                 print(f"[ERROR] Image {img_id}: Invalid label {label} found. Expected range [1, 145].")

        if not valid_boxes:
            target = {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "labels": torch.empty(0, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
            }
        else:
            boxes_tensor = torch.as_tensor(valid_boxes, dtype=torch.float32).reshape(-1, 4)
            boxes_tensor[:, 2:] += boxes_tensor[:, :2]
            target = {
                "boxes": boxes_tensor,
                "labels": valid_labels,
                "image_id": torch.tensor([img_id]),
            }

        if self.transforms is not None:
            transformed = self.transforms(
                image=np.array(img),
                bboxes=target["boxes"],
                labels=target["labels"],
            )
            img = transformed["image"]
            # 변환 후 박스가 비어있을 수 있음
            if len(transformed["bboxes"]) > 0:
                target["boxes"] = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
                target["labels"] = torch.as_tensor(transformed["labels"], dtype=torch.int64)
            else:
                target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                target["labels"] = torch.empty(0, dtype=torch.int64)
        
        if not isinstance(target["labels"], torch.Tensor):
            target["labels"] = torch.as_tensor(target["labels"], dtype=torch.int64)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)
