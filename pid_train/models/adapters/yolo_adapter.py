# pid_train/models/adapters/yolo_adapter.py
from torch import nn

from ..base_wrapper import UnifiedDetectionModel

class YOLOAdapter(nn.Module, UnifiedDetectionModel):
    """YOLO 모델용 어댑터"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.forward_train(images, targets)
        return self.forward_inference(images)
    
    def forward_train(self, images, targets):
        # YOLO 형식으로 변환
        yolo_targets = self.convert_targets(targets)
        loss_dict = self.model(images, yolo_targets)
        return loss_dict
    
    def forward_inference(self, images):
        results = self.model(images)
        return self.convert_predictions(results)
    
    def convert_targets(self, targets): # TODO: 아직 구현 다 안됨.
        """Torchvision 형식 → YOLO 형식"""
        # [image_idx, class, x_center, y_center, w, h] 형식으로 변환
        yolo_targets = []
        for img_idx, target in enumerate(targets):
            boxes = target['boxes']  # xyxy
            labels = target['labels']
            # xyxy → xywh (normalized) 변환 로직
            # ...
        return yolo_targets
    
    def convert_predictions(self, outputs):
        """YOLO 출력 → Torchvision 형식"""
        predictions = []
        for result in outputs:
            pred = {
                'boxes': result.xyxy[0],
                'labels': result.xyxy[0][:, 5].long(),
                'scores': result.xyxy[0][:, 4]
            }
            predictions.append(pred)
        return predictions