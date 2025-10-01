# pid_train/models/adapters/torchvision_adapter.py
import torch
import torch.nn as nn

from ..base_wrapper import UnifiedDetectionModel


class TorchvisionAdapter(nn.Module, UnifiedDetectionModel):
    """Torchvision 모델용 어댑터"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.forward_train(images, targets)
        return self.forward_inference(images)
    
    def forward_train(self, images, targets):
        # Torchvision은 그대로 전달
        return self.model(images, targets)
    
    def forward_inference(self, images):
        self.model.eval()
        with torch.no_grad():
            return self.model(images)
    
    def convert_targets(self, targets):
        return targets
    
    def convert_predictions(self, outputs):
        return outputs