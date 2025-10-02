from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from effdet import DetBenchPredict, unwrap_bench
from effdet.factory import create_model
from omegaconf import OmegaConf # OmegaConf를 import합니다.

from ..base_wrapper import UnifiedDetectionModel


class EffDetAdapter(nn.Module, UnifiedDetectionModel):
    """
    EfficientDet (effdet 라이브러리) 모델용 어댑터.
    UnifiedDetectionModel 인터페이스를 준수합니다.
    """

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, gradient_checkpointing: bool = False, **kwargs):
        """
        Args:
            model_name (str): 생성할 efficientdet 모델의 이름 (예: 'tf_efficientdet_d1').
            num_classes (int): 데이터셋의 클래스 수.
            pretrained (bool): 사전 학습된 가중치를 사용할지 여부.
            gradient_checkpointing (bool): Gradient Checkpointing 활성화 여부.
            **kwargs: `effdet.create_model`에 전달할 추가 인자.
        """
        super().__init__()
        
        # 1. 학습/검증용 모델(DetBenchTrain) 생성
        self.model_train = create_model(
            model_name,
            bench_task='train',
            num_classes=num_classes,
            pretrained=pretrained,
            **kwargs,
        )
        
        # --- gradient_checkpointing 처리 로직 추가 ---
        if gradient_checkpointing:
            # effdet 모델의 config는 기본적으로 읽기 전용(read-only)입니다.
            # 설정을 변경하기 위해 쓰기 가능 상태로 일시적으로 변경합니다.
            OmegaConf.set_readonly(self.model_train.config, False)
            # act_checkpointing을 활성화합니다.
            self.model_train.config.act_checkpointing = True
            # 다시 읽기 전용으로 설정하여 의도치 않은 변경을 방지합니다.
            OmegaConf.set_readonly(self.model_train.config, True)
            print("[INFO] Gradient Checkpointing enabled for EfficientDet.")
            
        # 2. 추론용 모델(DetBenchPredict) 생성 및 가중치 공유
        self.model_predict = DetBenchPredict(unwrap_bench(self.model_train))
    
    def forward(
        self, 
        images: torch.Tensor, 
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided during training.")
            return self.forward_train(images, targets)
        else:
            return self.forward_inference(images, targets)

    def forward_train(
        self, 
        images: torch.Tensor, 
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        converted_targets = self._convert_targets(targets, images.device)
        return self.model_train(images, converted_targets)

    def forward_inference(
        self, 
        images: torch.Tensor, 
        targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        if targets is not None:
            converted_targets = self._convert_targets(targets, images.device)
            output = self.model_train(images, converted_targets)
            output['predictions'] = self.convert_predictions(output['detections'])
            del output['detections']
            return output
        else:
            detections = self.model_predict(images)
            return self.convert_predictions(detections)

    def _convert_targets(self, targets: List[Dict[str, torch.Tensor]], device: torch.device) -> Dict[str, torch.Tensor]:
        boxes = [t['boxes'] for t in targets]
        labels = [t['labels'] for t in targets]
        
        bboxes_yxyx = [
            torch.stack([b[:, 1], b[:, 0], b[:, 3], b[:, 2]], dim=1) if len(b) > 0 else b
            for b in boxes
        ]

        padded_bboxes = self._collate_tensor_list(bboxes_yxyx, pad_value=0)
        padded_labels = self._collate_tensor_list(labels, pad_value=-1)
        
        default_img_size = torch.tensor([512., 512.], device=device)
        default_img_scale = torch.tensor(1.0, device=device)

        img_sizes = torch.stack([t.get('img_size', default_img_size) for t in targets]).to(device)
        img_scales = torch.stack([t.get('img_scale', default_img_scale) for t in targets]).to(device)

        return {
            'bbox': padded_bboxes,
            'cls': padded_labels,
            'img_size': img_sizes,
            'img_scale': img_scales
        }
        
    def convert_predictions(self, detections: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        predictions = []
        for i in range(detections.shape[0]):
            img_detections = detections[i]
            valid_mask = img_detections[:, 4] >= 0
            img_detections = img_detections[valid_mask]
            predictions.append({
                "boxes": img_detections[:, :4],
                "scores": img_detections[:, 4],
                "labels": img_detections[:, 5].int(),
            })
        return predictions
    
    @staticmethod
    def _collate_tensor_list(batch: List[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
        max_len = max(len(x) for x in batch) if batch else 0
        if max_len == 0:
            shape_suffix = batch[0].shape[1:] if batch and len(batch[0].shape) > 1 else ()
            return torch.empty((len(batch), 0, *shape_suffix), dtype=torch.float32)

        padded_batch = torch.full(
            (len(batch), max_len, *batch[0].shape[1:]),
            float(pad_value),
            dtype=batch[0].dtype,
            device=batch[0].device,
        )
        for i, x in enumerate(batch):
            if len(x) > 0:
                padded_batch[i, : len(x)] = x
        return padded_batch