import hydra
import torch
import effdet
from lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from omegaconf import DictConfig

from pid_train.models.base_wrapper import UnifiedDetectionModel


class ObjectDetectionLitModule(LightningModule):
    """
    Object Detection을 위한 LightningModule.
    Hydra를 통해 모델, 옵티마이저, 스케줄러 설정을 주입받습니다.
    """
    def __init__(
        self,
        model: UnifiedDetectionModel,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
    ):
        super().__init__()
        
        self.model = model
        self.save_hyperparameters(ignore=['model']) 

        # 평가지표: mAP
        self.val_map = MeanAveragePrecision(box_format='xyxy', class_metrics=True)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        self.log_dict(loss_dict, prog_bar=True)
        self.log("train/total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        
        predictions = self.model(images, targets)

        self.val_map.update(predictions, targets)

    def on_validation_epoch_end(self):
        """검증 에폭이 끝날 때 호출됩니다."""
        map_metrics = self.val_map.compute()

        map_per_class = map_metrics.pop("map_per_class", None)
        mar_100_per_class = map_metrics.pop("mar_100_per_class", None)
        classes = map_metrics.pop("classes", None)

        self.log_dict({f"val/{k}": v for k, v in map_metrics.items()}, prog_bar=True)

        if map_per_class is not None and isinstance(map_per_class, torch.Tensor) and map_per_class.ndim > 0:
            for i, ap in enumerate(map_per_class):
                if ap != -1:
                    self.log(f"val_map_class/class_{i}", ap)
        
        self.val_map.reset()

    def configure_optimizers(self):
        """옵티마이저와 스케줄러를 설정합니다."""
        optimizer_cfg = self.hparams.optimizer_cfg.copy()
        optimizer_cfg._target_ = optimizer_cfg.pop('class')
        optimizer_cfg.params = self.model.parameters()
        optimizer = hydra.utils.instantiate(optimizer_cfg)

        scheduler_cfg = self.hparams.scheduler_cfg.copy()
        scheduler_cfg._target_ = scheduler_cfg.pop('class')
        scheduler = hydra.utils.instantiate(scheduler_cfg, optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }