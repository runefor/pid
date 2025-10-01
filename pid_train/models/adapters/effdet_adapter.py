# pid_train/models/adapters/effdet_adapter.py
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from ..base_wrapper import UnifiedDetectionModel


class EffDetAdapter(nn.Module, UnifiedDetectionModel):
    """EfficientDet (effdet 라이브러리) 모델용 어댑터"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model  # DetBenchTrain
    
    def forward(self, images: torch.Tensor, targets: Optional[List[Dict]] = None):
        if self.training and targets is not None:
            return self.forward_train(images, targets)
        return self.forward_inference(images, targets)
    
    def forward_train(self, images: torch.Tensor, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """Training 모드"""
        # 타겟 변환 (필요시)
        converted_targets = self.convert_targets(targets)
        return self.model(images, converted_targets)
    
    def forward_inference(self, images: torch.Tensor, targets: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        """Inference 모드"""
        self.model.eval()
        converted_targets = self.convert_targets(targets)
        
        # 디버깅: effdet 직전에 다시 확인
        print(f"[EffDetAdapter] Before effdet call:")
        for i, t in enumerate(converted_targets):
            print(f"  Target {i}: {t.keys()}, label_num_positives={t.get('label_num_positives', 'MISSING')}")
        
        with torch.no_grad():
            predictions = self.model(images, converted_targets)
        return predictions
    

    def convert_targets(self, targets: List[Dict]) -> List[Dict]:
        """Torchvision 형식 → EfficientDet 형식"""
        converted = []
        
        print(f"[EffDetAdapter] Converting {len(targets)} targets")
        
        for i, t in enumerate(targets):
            print(f"[EffDetAdapter] Target {i} keys: {t.keys()}")
            
            # 이미 effdet 형식인 경우
            if 'bbox' in t and 'cls' in t and 'label_num_positives' in t:
                print(f"[EffDetAdapter] Target {i} already in effdet format")
                converted.append(t)
            # torchvision 형식인 경우 변환
            elif 'boxes' in t and 'labels' in t:
                print(f"[EffDetAdapter] Target {i} converting from torchvision format")
                new_target = {
                    'bbox': t['boxes'],
                    'cls': t['labels'],
                    'label_num_positives': torch.tensor([len(t['boxes'])], dtype=torch.int64, device=t['boxes'].device)
                }
                print(f"[EffDetAdapter] Converted target {i} keys: {new_target.keys()}")
                converted.append(new_target)
            else:
                raise ValueError(f"Unknown target format. Keys: {t.keys()}")
        
        print(f"[EffDetAdapter] Converted {len(converted)} targets")
        print(f"[EffDetAdapter] First converted target keys: {converted[0].keys()}")
        
        return converted
    
    def convert_predictions(self, outputs: Any) -> List[Dict[str, torch.Tensor]]:
        """EfficientDet 출력 → Torchvision 형식"""
        return outputs