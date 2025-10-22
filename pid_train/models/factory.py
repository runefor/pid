import torch
import torchvision
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from .protocols import ObjectDetectionModelProtocol
from .architecture import ObjectDetector, EfficientDetTrainWrapper
from .base_wrapper import UnifiedDetectionModel
from .adapters import TorchvisionAdapter, YOLOAdapter, EffDetAdapter


def create_model(
    framework: str,  # 'torchvision', 'yolo', 'mmdet'
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    gradient_checkpointing: bool = False,
    **kwargs,
) -> UnifiedDetectionModel:
    """프레임워크에 맞는 어댑터로 모델 생성"""
    if framework == 'torchvision':
        base_model = _create_torchvision_model(model_name, num_classes, pretrained, gradient_checkpointing)
        return TorchvisionAdapter(base_model)
    
    elif framework == 'yolo':
        base_model = _create_yolo_model(num_classes, pretrained, model_name, gradient_checkpointing)
        return YOLOAdapter(base_model)
    
    elif framework == 'effdet':
        return EffDetAdapter(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            gradient_checkpointing=gradient_checkpointing, # 어댑터 내부에서 처리하도록 전달
            **kwargs,
        )
    
    else:
        raise ValueError(f"Unknown framework: {framework}")

# --- 팩토리 함수들 ---

def _create_torchvision_model(model_name: str, num_classes: int, pretrained: bool = True, gradient_checkpointing: bool = False):
    """Torchvision 모델들 - 각각 다른 처리 필요"""
    
    # 모델 타입별로 분기
    if 'fasterrcnn' in model_name:
        return create_faster_rcnn(num_classes, pretrained, model_name, gradient_checkpointing)
    
    elif 'retinanet' in model_name:
        return create_retinanet(num_classes, pretrained, model_name, gradient_checkpointing)
    
    elif 'ssd' in model_name:
        return create_ssd(num_classes, pretrained, model_name, gradient_checkpointing)
    
    # elif 'fcos' in model_name:
    #     return create_fcos(num_classes, pretrained, model_name, **kwargs)
    
    else:
        raise ValueError(f"Unknown torchvision model: {model_name}")

def create_faster_rcnn(num_classes: int, pretrained=True, model_name: str = "fasterrcnn_resnet50_fpn", gradient_checkpointing: bool = False) -> ObjectDetectionModelProtocol:
    """
    Torchvision의 pre-trained Faster R-CNN 모델을 생성합니다.
    """
    model_fn = getattr(detection_models, model_name)
    model_impl: detection_models.FasterRCNN = model_fn(weights='DEFAULT' if pretrained else None)

    in_features = model_impl.roi_heads.box_predictor.cls_score.in_features
    model_impl.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if gradient_checkpointing:
        return ObjectDetector(model_impl, gradient_checkpointing=True)
    return model_impl


def create_retinanet(num_classes: int, pretrained=True, model_name: str = "retinanet_resnet50_fpn_v2", gradient_checkpointing: bool = False) -> ObjectDetectionModelProtocol:
    """Torchvision의 pre-trained RetinaNet 모델을 생성합니다."""
    
    model_fn = getattr(detection_models, model_name)
    
    model_impl: detection_models.RetinaNet = model_fn(weights='DEFAULT' if pretrained else None)
    
    num_anchors = model_impl.head.classification_head.num_anchors
    in_channels = model_impl.backbone.out_channels
    
    # Classification head 재생성
    model_impl.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )
    
    if gradient_checkpointing:
        return ObjectDetector(model_impl, gradient_checkpointing=True)
    return model_impl

def create_ssd(num_classes: int, pretrained=True, model_name: str = "ssdlite320_mobilenet_v3_large", gradient_checkpointing: bool = False) -> ObjectDetectionModelProtocol:
    """
    Torchvision의 pre-trained SSD 모델을 생성합니다.
    """
    # from torchvision.models.detection.ssd import SSDClassificationHead
    model_fn = getattr(detection_models, model_name)
    model_impl: detection_models.ssd.SSD = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights='DEFAULT' if pretrained else None, 
            num_classes=num_classes
        )
    
    if gradient_checkpointing:
        return ObjectDetector(model_impl, gradient_checkpointing=True)
    
    return model_impl

def _create_yolo_model(num_classes: int, pretrained=True, model_name='yolov8s.pt', gradient_checkpointing: bool = False) -> ObjectDetectionModelProtocol:
    """
    ultralytics 라이브러리를 통해 YOLOv8 모델을 로드합니다.
    """
    from ultralytics import YOLO
    model_impl = YOLO(model_name)
    
    # 모델의 클래스 수를 업데이트해야 할 경우,
    # 이 부분은 YOLO 모델의 구조에 따라 다를 수 있습니다.
    # model_impl.model.yaml['nc'] = num_classes

    return model_impl




# ==================================
# 잘 동작하지 않는 모델들
# ==================================

# BUG: effdet 라이브러리 문제로 제대로 동작하지 않음.
def create_efficientdet(num_classes: int, pretrained=True, model_name='tf_efficientdet_d0', gradient_checkpointing: bool = False):
    """
    effdet 라이브러리를 사용하여 전이 학습을 위한 EfficientDet 모델을 생성합니다.
    """
    import effdet
    from omegaconf import OmegaConf
    
    model = effdet.create_model(
        model_name,
        bench_task='train',
        pretrained=pretrained,
    )
    model.model.reset_head(num_classes=num_classes)
    
    if gradient_checkpointing:
        OmegaConf.set_readonly(model.model.config, False)
        model.model.config.act_checkpointing = True
        OmegaConf.set_readonly(model.model.config, True)
        print("[INFO] Gradient Checkpointing enabled for EfficientDet.")

    return model