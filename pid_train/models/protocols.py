from typing import Dict, List, Optional, Protocol, Tuple, overload

import torch


class ObjectDetectionModelProtocol(Protocol):
    """
    이 프로젝트에서 사용하는 모든 Object Detection 모델이 따라야 하는 프로토콜.
    Torchvision의 detection 모델 인터페이스를 따릅니다.
    """
    # 모든 nn.Module은 training 속성을 가지고 있으므로, 이를 명시해줍니다.
    training: bool

    # @overload를 사용하여 학습/추론 시의 시그니처를 명확히 구분합니다.
    # IDE가 현재 모드(training/eval)에 따라 올바른 입출력 타입을 추론하는 데 도움을 줍니다.

    @overload
    def __call__(
        self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """학습 시 호출 시그니처: images와 targets를 받고, loss 딕셔너리를 반환"""
        ...

    @overload
    def __call__(
        self, images: List[torch.Tensor], targets: None = None
    ) -> List[Dict[str, torch.Tensor]]:
        """추론 시 호출 시그니처: images만 받고, 예측 결과 리스트를 반환"""
        ...

    def __call__(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ):
        """프로토콜의 실제 구현부. 내용은 중요하지 않습니다."""
        raise NotImplementedError