import torch.nn as nn
from .protocols import ObjectDetectionModelProtocol

class ObjectDetector(nn.Module, ObjectDetectionModelProtocol):
    """
    실제 모델을 감싸는 Wrapper 클래스.
    Gradient Checkpointing과 같은 추가적인 기능을 적용하기 용이합니다.
    """
    def __init__(self, model: nn.Module, gradient_checkpointing: bool = False):
        super().__init__()
        self.model = model
        self.gradient_checkpointing = gradient_checkpointing

        if self.gradient_checkpointing:
            # PyTorch Lightning이 이 속성을 보고 그래디언트 체크포인팅을 활성화합니다.
            # 단, 모델의 모든 서브모듈이 이를 지원해야 효과가 있습니다.
            # 복잡한 모델의 경우, 특정 부분(예: backbone)에만 적용해야 할 수도 있습니다.
            self.model.gradient_checkpointing = True
            print("[INFO] Gradient Checkpointing enabled for the model.")

    def forward(self, images, targets=None):
        # 학습과 추론 시 모델의 동작이 다르므로 그대로 전달합니다.
        return self.model(images, targets)
