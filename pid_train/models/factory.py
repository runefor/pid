import timm
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .protocols import ObjectDetectionModelProtocol
from .architecture import ObjectDetector

# --- 팩토리 함수들 ---

def create_faster_rcnn(num_classes, pretrained=True, gradient_checkpointing: bool = False) -> ObjectDetectionModelProtocol:
    """
    Torchvision의 pre-trained Faster R-CNN 모델을 생성하고,
    분류기(classifier)만 주어진 클래스 수에 맞게 교체합니다.
    """
    model_impl = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT' if pretrained else None)

    # Predictor 교체
    in_features = model_impl.roi_heads.box_predictor.cls_score.in_features
    model_impl.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Wrapper 클래스로 모델을 감싸서 반환
    model = ObjectDetector(model_impl, gradient_checkpointing=gradient_checkpointing)
    return model


def create_efficientdet(num_classes, pretrained=True, model_name='tf_efficientdet_d0', gradient_checkpointing: bool = False) -> ObjectDetectionModelProtocol:
    """
    Timm 라이브러리의 pre-trained EfficientDet 모델을 생성합니다.
    """
    model_impl = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        gradient_checkpointing=gradient_checkpointing  # timm 모델은 이 플래그를 직접 지원
    )
    model = ObjectDetector(model_impl, gradient_checkpointing=False) # timm에서 이미 처리
    return model


def create_yolo(model_name='yolov5s', pretrained=True, gradient_checkpointing: bool = False) -> ObjectDetectionModelProtocol:
    """
    PyTorch Hub를 통해 YOLOv5 모델을 로드합니다.
    """
    model_impl = torch.hub.load(
        'ultralytics/yolov5', 
        model_name,
        pretrained=pretrained
    )
    # YOLO 모델은 gradient checkpointing을 내부적으로 지원하지 않을 수 있음
    model = ObjectDetector(model_impl, gradient_checkpointing=gradient_checkpointing)
    return model
