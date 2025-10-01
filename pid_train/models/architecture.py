import torch
import torch.nn as nn
from .protocols import ObjectDetectionModelProtocol

class ObjectDetector(nn.Module):
    """
    모델에 공통 기능(gradient checkpointing 등)을 추가하는 경량 wrapper
    """
    def __init__(self, model: nn.Module, gradient_checkpointing: bool = False):
        super().__init__()
        self.model = model
        
        if gradient_checkpointing:
            self._apply_gradient_checkpointing()

    def _apply_gradient_checkpointing(self):
        """모델에 gradient checkpointing 적용"""
        try:
            # Torchvision 모델의 경우 backbone에 적용
            if hasattr(self.model, 'backbone'):
                if hasattr(self.model.backbone, 'body'):
                    # ResNet 등
                    for module in self.model.backbone.body.modules():
                        if isinstance(module, nn.Sequential):
                            module.gradient_checkpointing = True
            
            # 일반적인 경우
            self.model.gradient_checkpointing = True
            print("[INFO] Gradient Checkpointing enabled.")
        except Exception as e:
            print(f"[WARNING] Could not apply gradient_checkpointing: {e}")

    def forward(self, images, targets=None):
        """모델 forward를 그대로 전달"""
        return self.model(images, targets)
    

class EfficientDetTrainWrapper(nn.Module):
    """DetBenchTrain을 validation에서도 사용할 수 있도록 래핑"""
    
    def __init__(self, model_train):
        super().__init__()
        self.model = model_train
        
    def forward(self, images, targets=None):
        if self.training:
            # Training 모드: targets 필수
            return self.model(images, targets)
        else:
            # Validation 모드: targets를 전달하되, 모델을 일시적으로 eval로 변경
            # DetBenchTrain은 eval 모드에서는 loss를 계산하지 않고 predictions만 반환
            self.model.eval()
            with torch.no_grad():
                # targets를 전달해야 하지만, eval 모드에서는 검증 없이 통과
                output = self.model(images, targets)
            return output