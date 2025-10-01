from typing import List
import os

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="pkg://pid_train/config", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Hydra 설정에 따라 데이터 모듈, 라이트닝 모듈, 트레이너를 인스턴스화하고
    학습을 시작하는 메인 함수.
    """
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