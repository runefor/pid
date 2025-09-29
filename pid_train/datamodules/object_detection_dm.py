from typing import Optional, List, Dict, Any, Tuple

import hydra
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    DataLoader에 사용될 collate_fn.
    배치를 이미지 텐서와 타겟 리스트로 변환합니다.
    """
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

class ObjectDetectionDataModule(LightningDataModule):
    """
    Object Detection을 위한 LightningDataModule.
    Hydra를 통해 데이터셋, 변환(transform), 로더(loader) 설정을 주입받습니다.
    """
    def __init__(
        self,
        dataset_cfg: DictConfig,
        transform_cfg: DictConfig,
        loader_cfg: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        학습/검증 데이터셋을 설정합니다.
        """
        # transform_cfg를 복사하고, class를 _target_으로 변경하여 인스턴스화
        transform_cfg = self.hparams.transform_cfg.copy()
        transform_cfg._target_ = transform_cfg.pop("class")
        train_transforms = hydra.utils.instantiate(transform_cfg)

        # 검증용 변환 파이프라인에서 HorizontalFlip 제거
        val_transform_cfg = transform_cfg.copy()
        val_transform_cfg.transforms.pop(1)
        val_transforms = hydra.utils.instantiate(val_transform_cfg)

        if stage == 'fit' or stage is None:
            # 학습 데이터셋 생성
            train_cfg = self.hparams.dataset_cfg.train.copy()
            train_cfg._target_ = train_cfg.pop("class")
            self.train_dataset = hydra.utils.instantiate(
                train_cfg,
                transforms=train_transforms
            )

            # 검증 데이터셋 생성
            val_cfg = self.hparams.dataset_cfg.val.copy()
            val_cfg._target_ = val_cfg.pop("class")
            self.val_dataset = hydra.utils.instantiate(
                val_cfg,
                transforms=val_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            **self.hparams.loader_cfg.train
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            **self.hparams.loader_cfg.val
        )