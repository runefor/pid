import hydra
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import albumentations as A
import torch

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

class ObjectDetectionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_cfg: DictConfig,
        transform_cfg: DictConfig,
        loader_cfg: DictConfig,
    ):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.transform_cfg = transform_cfg
        self.loader_cfg = loader_cfg

    def setup(self, stage=None):
        # Transform 생성
        transforms_list = []
        for t in self.transform_cfg.transforms:
            # t가 DictConfig인지 확인
            if isinstance(t, DictConfig):
                transforms_list.append(hydra.utils.instantiate(t))
            else:
                # 이미 인스턴스화된 경우
                transforms_list.append(t)
        
        # bbox_params 인스턴스화
        if isinstance(self.transform_cfg.bbox_params, DictConfig):
            bbox_params = hydra.utils.instantiate(self.transform_cfg.bbox_params)
        else:
            bbox_params = self.transform_cfg.bbox_params
        
        # albumentations.Compose로 조합
        transforms = A.Compose(transforms_list, bbox_params=bbox_params)
        
        # Dataset 클래스 가져오기
        train_dataset_class = hydra.utils.get_class(self.dataset_cfg.train['class'])
        val_dataset_class = hydra.utils.get_class(self.dataset_cfg.val['class'])
        
        self.train_dataset = train_dataset_class(
            image_dir=self.dataset_cfg.train.image_dir,
            annotation_file=self.dataset_cfg.train.annotation_file,
            transforms=transforms,
            tile_size=self.dataset_cfg.tile_size,
            overlap=self.dataset_cfg.overlap,
        )
        
        self.val_dataset = val_dataset_class(
            image_dir=self.dataset_cfg.val.image_dir,
            annotation_file=self.dataset_cfg.val.annotation_file,
            transforms=transforms,
            tile_size=self.dataset_cfg.tile_size,
            overlap=self.dataset_cfg.overlap,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            **self.loader_cfg.train
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            **self.loader_cfg.val
        )