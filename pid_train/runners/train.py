from typing import List
import os
import yaml

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

@hydra.main(version_base=None, config_path="pkg://pid_train/config", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra 설정에 따라 데이터 모듈, 라이트닝 모듈, 트레이너를 인스턴스화하고
    학습을 시작하는 메인 함수.
    """
    if cfg.models.framework == 'yolo':
        print("Starting YOLO training!")
        
        # 1. Create dataset yaml file
        dataset_yaml_content = {
            'train': os.path.abspath(cfg.dataset.train.image_dir),
            'val': os.path.abspath(cfg.dataset.val.image_dir),
            'nc': cfg.models.num_classes,
            'names': [f'class_{i}' for i in range(cfg.models.num_classes)]
        }
        
        dataset_yaml_path = 'yolo_dataset.yaml'
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_yaml_content, f)

        # 2. Instantiate YOLO model
        model = YOLO(cfg.models.model_name)

        # 3. Start training
        model.train(
            data=dataset_yaml_path,
            epochs=cfg.trainer.max_epochs,
            batch=cfg.loader_cfg.train.batch_size,
            imgsz=cfg.dataset.tile_size,
        )
        print("YOLO training finished!")

    else:
        # --- 1. 인스턴스화 ---
        print("Instantiating datamodule...")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

        print("Instantiating lightning module...")
        lit_module: LightningModule = hydra.utils.instantiate(cfg.lit_module)
        
        # --- 로거 인스턴스화 ---
        loggers: List[Logger] = []
        if "logging" in cfg:
            print("Instantiating loggers...")
            for logger_name, logger_cfg in cfg.logging.items():
                print(f" - Instantiating logger: {logger_name}")
                logger: Logger = hydra.utils.instantiate(logger_cfg)
                loggers.append(logger)

        print("Instantiating trainer...")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

        # --- 2. 학습 시작 ---
        print("Starting training!")
        trainer.fit(model=lit_module, datamodule=datamodule)

        print("Training finished!")


if __name__ == "__main__":
    main()
