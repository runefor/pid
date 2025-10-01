# Gemini Code Understanding

## Project Overview

This project is a machine learning pipeline for object detection. It uses PyTorch Lightning for training and Hydra for configuration management. The pipeline is divided into several stages:

*   **Preprocessing:** The `main_preprocess_pipeline.py` script loads data, adds features, and performs exploratory data analysis (EDA).
*   **Training:** The `pid_train.py` script, which calls `pid_train/runners/train.py`, trains an object detection model. The training process is highly configurable using Hydra.
*   **Prediction:** The `predict.py` script is likely used for making predictions with a trained model.
*   **Evaluation:** The `evaluate.py` script is likely used for evaluating the performance of a trained model.
*   **MMDetection Training:** The `pid_mm_train/` directory contains scripts for training models using the MMDetection framework.

The project has recently been refactored to support a more modular architecture. It now uses a model factory with adapters for different object detection frameworks, including:

*   **Torchvision**
*   **YOLO**
*   **EfficientDet (effdet)**

The project also includes support for **tiled datasets**, which can be useful for training on large images.

The project uses a variety of open-source libraries, including:

*   **PyTorch Lightning:** A lightweight PyTorch wrapper for high-performance AI research.
*   **Hydra:** A framework for elegantly configuring complex applications.
*   **MLflow:** An open source platform for the machine learning lifecycle.
*   **EfficientDet:** A scalable and efficient object detection model.
*   **Albumentations:** A fast and flexible library for image augmentations.
*   **Dask:** A flexible library for parallel computing in Python.
*   **MMDetection:** An open source object detection toolbox based on PyTorch.

## Building and Running

### 1. Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

### 2. Preprocessing

To run the preprocessing pipeline, execute the following command:

```bash
python main_preprocess_pipeline.py
```

### 3. Training

To train the model, you can use the `pid_train.py` script. The training process is configured using Hydra. The main configuration file is `pid_train/config/train.yaml`. This file allows you to select the dataset, model, and other training parameters.

To start training with the default configuration, run:

```bash
python pid_train.py
```

You can override the default configuration from the command line. For example, to train with a different model, you can modify the `models` parameter. The new model factory allows you to specify the framework and model name.

For example, to train a RetinaNet model, you would modify the `pid_train/config/train.yaml` to select the `retinanet` model configuration:

```yaml
# pid_train/config/train.yaml
...
defaults:
  - base.yaml
  - dataset: config_torchvision
  - models: retinanet
  - _self_
...
```

And the `pid_train/config/models/retinanet.yaml` would look like this:

```yaml
# pid_train/config/models/retinanet.yaml
_target_: pid_train.models.factory.create_model
framework: torchvision
model_name: "retinanet_resnet50_fpn_v2"
num_classes: 146
pretrained: true
gradient_checkpointing: true
```

### 4. Prediction and Evaluation

The `predict.py` and `evaluate.py` scripts are likely used for prediction and evaluation, respectively. The exact usage of these scripts is not immediately clear from the code, but they likely take a trained model and a dataset as input.

## Development Conventions

*   **Configuration:** The project uses Hydra for configuration. All configurations are stored in the `pid_train/config` directory. The configuration is now more modular, with separate files for datasets (e.g., `pid_train/config/dataset/config_torchvision.yaml`) and models (e.g., `pid_train/config/models/faster_rcnn.yaml`).
*   **Training:** The project uses PyTorch Lightning for training. The training logic is encapsulated in `LightningModule` and `LightningDataModule` classes.
*   **Logging:** The project uses TensorBoard and MLflow for logging.
*   **Data:** The project seems to handle large datasets, as indicated by the use of Dask. The data is stored in the `assets` directory. The project now supports tiled datasets, with base classes and implementations in the `pid_train/datasets` directory.
*   **Code Structure:** The code is organized into several directories. A major change is the introduction of the `pid_train/models/adapters` directory, which holds the adapter classes for different object detection frameworks. The `pid_train/models/factory.py` contains the model creation logic. The `pid_mm_train/` directory is a new addition for MMDetection experiments.