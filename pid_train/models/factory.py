import timm
import torch
import torchvision
import effdet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from omegaconf import OmegaConf

from .protocols import ObjectDetectionModelProtocol
from .architecture import ObjectDetector

# --- 팩토리 함수들 ---

def create_faster_rcnn(num_classes, pretrained=True, gradient_checkpointing: bool = False, val_requires_targets: bool = False) -> ObjectDetectionModelProtocol:
    """
    Torchvision의 pre-trained Faster R-CNN 모델을 생성합니다.
    """
    model_impl = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)

    in_features = model_impl.roi_heads.box_predictor.cls_score.in_features
    model_impl.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = ObjectDetector(
        model_impl, 
        gradient_checkpointing=gradient_checkpointing, 
        val_requires_targets=val_requires_targets
    )
    return model


def create_efficientdet(num_classes, pretrained=True, model_name='tf_efficientdet_d0', gradient_checkpointing: bool = False, val_requires_targets: bool = False) -> ObjectDetectionModelProtocol:
    """
    effdet 라이브러리를 사용하여 전이 학습을 위한 EfficientDet 모델을 생성합니다.
    """
    # 1. 사전 학습된 모델을 원본 클래스 수 그대로 먼저 생성합니다.
    model_impl = effdet.create_model(
        model_name,
        bench_task='train',
        pretrained=pretrained,
    )

    # 2. 모델의 head(최종 분류기)를 우리의 클래스 수에 맞게 리셋합니다.
    model_impl.model.reset_head(num_classes=num_classes)
    
    # 3. Gradient Checkpointing 설정
    if gradient_checkpointing:
        # 모델의 내부 config는 읽기 전용이므로, 상태를 변경하고 다시 잠급니다.
        OmegaConf.set_readonly(model_impl.model.config, False)
        model_impl.model.config.act_checkpointing = True
        OmegaConf.set_readonly(model_impl.model.config, True)
        print("[INFO] Gradient Checkpointing enabled for EfficientDet.")

    # 일관성을 위해 ObjectDetector로 감싸서 반환
    return ObjectDetector(
        model_impl, 
        gradient_checkpointing=False, # effdet 모델은 내부 config로 제어됨
        val_requires_targets=val_requires_targets
    )


def create_yolo(model_name='yolov5s', pretrained=True, gradient_checkpointing: bool = False, val_requires_targets: bool = False) -> ObjectDetectionModelProtocol:
    """
    PyTorch Hub를 통해 YOLOv5 모델을 로드합니다.
    """
    model_impl = torch.hub.load(
        'ultralytics/yolov5', 
        model_name,
        pretrained=pretrained
    )
    model = ObjectDetector(
        model_impl, 
        gradient_checkpointing=gradient_checkpointing,
        val_requires_targets=val_requires_targets
    )
    return model
