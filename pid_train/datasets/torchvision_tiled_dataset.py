# pid_train/datasets/torchvision_tiled_dataset.py
from typing import Dict, List
import torch
from .base_tiled_dataset import BaseTiledDataset

class TorchvisionTiledDataset(BaseTiledDataset):
    """Torchvision 모델용 (Faster R-CNN, RetinaNet 등)"""
    
    def _create_target(self, boxes: List, labels: List, image_id: int) -> Dict[str, torch.Tensor]:
        return {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([image_id])
        }