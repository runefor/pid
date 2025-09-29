import torch.nn as nn
from .protocols import ObjectDetectionModelProtocol

class ObjectDetector(nn.Module, ObjectDetectionModelProtocol):
    """
    실제 모델을 감싸는 Wrapper 클래스.
    추가적인 속성이나 기능을 적용하기 용이합니다.
    """
    def __init__(self, model: nn.Module, gradient_checkpointing: bool = False, val_requires_targets: bool = False):
        super().__init__()
        self.model = model
        self.gradient_checkpointing = gradient_checkpointing
        self.val_requires_targets = val_requires_targets # 플래그 저장

        if self.gradient_checkpointing:
            # PyTorch Lightning이 이 속성을 보고 그래디언트 체크포인팅을 활성화합니다.
            # 복잡한 모델의 경우, 특정 부분(예: backbone)에만 적용해야 할 수도 있습니다.
            try:
                self.model.gradient_checkpointing = True
                print("[INFO] Gradient Checkpointing enabled for the model.")
            except Exception as e:
                print(f"[WARNING] Could not apply gradient_checkpointing automatically: {e}")

    def forward(self, images, targets=None):
        return self.model(images, targets)