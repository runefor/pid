import hydra
import numpy as np
import torch
import effdet
import seaborn as sns
import matplotlib.pyplot as plt
from lightning import LightningModule
from omegaconf import DictConfig

from pid_train.models.base_wrapper import UnifiedDetectionModel
from pid_train.metrics.eval_metrics import ObjectDetectionMetrics, ConfusionMatrix, ObjectDetectionAP


from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger

import tempfile
import os
import time

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
        dataset_cfg: DictConfig,
        num_classes: int,
        max_dets: int = 100,
    ):
        super().__init__()
        
        self.model = model
        # self.model = torch.compile(model)
        self.save_hyperparameters(ignore=['model']) 

        # 평가지표
        self.object_detection_metrics = ObjectDetectionMetrics(num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes)
        self.object_detection_ap = ObjectDetectionAP()

        # 검증 단계의 출력을 저장하기 위한 리스트
        self.validation_step_outputs = []

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
        self.validation_step_outputs.append({'preds': predictions, 'targets': targets})

    def on_validation_epoch_end(self):
        """검증 에폭이 끝날 때 호출됩니다."""
        if not self.validation_step_outputs:
            return

        # 모든 예측과 정답을 하나의 리스트로 평탄화합니다.
        all_preds = []
        all_targets = []
        for output in self.validation_step_outputs:
            all_preds.extend(output['preds'])
            all_targets.extend(output['targets'])

        # --- 평가지표 계산 시간 측정 시작 ---
        start_time = time.time()

        # F1 Score, Confusion Matrix는 매 에폭 계산
        self.object_detection_metrics.update(all_preds, all_targets)
        self.confusion_matrix.update(all_preds, all_targets)

        # ObjectDetectionMetrics 로깅
        obj_det_metrics = self.object_detection_metrics.compute()
        per_class_f1 = obj_det_metrics.pop("per_class_f1")
        self.log_dict({f"val/{k}": v for k, v in obj_det_metrics.items()}, prog_bar=True)
        for i, f1 in enumerate(per_class_f1):
            self.log(f"val_f1_class/class_{i}", f1)
        self.object_detection_metrics.reset()

        # ConfusionMatrix 로깅
        confusion_matrix = self.confusion_matrix.compute()
        self.log("val/confusion_matrix_mean", confusion_matrix.mean())

        # mAP는 마지막 에폭에서만 계산
        is_last_epoch = (self.current_epoch == self.trainer.max_epochs - 1)
        if is_last_epoch:
            print("\nCalculating mAP for the final validation epoch...")
            self.object_detection_ap.update(all_preds, all_targets)
            obj_det_ap_metrics = self.object_detection_ap.compute()
            for k, v in obj_det_ap_metrics.items():
                if k == 'classes':
                    continue
                if hasattr(v, '__len__') and v.dim() > 0 and len(v) > 1:
                    for i, val in enumerate(v):
                        self.log(f"val/{k}_{i}", val)
                else:
                    self.log(f"val/{k}", v)
            self.object_detection_ap.reset()

        # --- 평가지표 계산 시간 측정 종료 및 로깅 ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.log("val/metric_computation_time", elapsed_time, prog_bar=True)

        # 저장된 출력 비우기
        self.validation_step_outputs.clear()

    def on_train_end(self):
        """학습이 끝난 후 호출됩니다."""
        confusion_matrix = self.confusion_matrix.compute().cpu().numpy()
        confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(confusion_matrix_normalized, annot=True, fmt='.0f', ax=ax, annot_kws={"size": 8})
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.add_figure("Confusion Matrix", fig, self.global_step)
            elif isinstance(logger, MLFlowLogger):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.savefig(tmpfile.name)
                    logger.experiment.log_artifact(logger.run_id, tmpfile.name, "confusion_matrix.png")
                os.remove(tmpfile.name)
        
        self.confusion_matrix.reset()

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