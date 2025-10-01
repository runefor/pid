# pid_train/datasets/effdet_tiled_dataset.py
from typing import Dict, List
import torch
from .base_tiled_dataset import BaseTiledDataset

# BUG: 라이브러리 오류인지 동작을 하지 않음. 일단 사용하지 않는 것이 좋을 것 같음.
class EffdetTiledDataset(BaseTiledDataset):
    """EfficientDet 모델용"""
    
    def _create_target(self, boxes: List, labels: List) -> Dict[str, torch.Tensor]:
        target = {
            'bbox': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            'cls': torch.as_tensor(labels, dtype=torch.int64),
            'label_num_positives': torch.tensor([len(boxes)], dtype=torch.int64)
        }
        
        return target