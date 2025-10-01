# pid_train/models/base_wrapper.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import torch

class UnifiedDetectionModel(ABC):
    """모든 detection 모델의 통합 인터페이스"""
    
    @abstractmethod
    def forward_train(self, images: torch.Tensor, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """Training forward - loss dict 반환"""
        pass
    
    @abstractmethod
    def forward_inference(self, images: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Inference forward - predictions 반환"""
        pass
    
    @abstractmethod
    def convert_targets(self, targets: List[Dict]) -> Any:
        """각 라이브러리의 타겟 형식으로 변환"""
        pass
    
    @abstractmethod
    def convert_predictions(self, outputs: Any) -> List[Dict[str, torch.Tensor]]:
        """각 라이브러리의 출력을 통일된 형식으로 변환"""
        pass