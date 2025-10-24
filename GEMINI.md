# Gemini Code Understanding: P&ID Object Detection Project

## 1. Project Overview

This project is a machine learning pipeline for object detection in P&ID (Piping and Instrumentation Diagram) drawings. It uses PyTorch Lightning for training, Hydra for configuration management, and a tiling-based approach to handle large, high-resolution images.

The pipeline has been significantly refactored to improve performance, maintainability, and consistency across different stages.

### Core Technologies
- **ML Framework**: PyTorch, PyTorch Lightning
- **Configuration**: Hydra
- **Data Handling**: Pandas, Dask, Numpy
- **Tiling & Augmentation**: Albumentations
- **Experiment Tracking**: MLflow, TensorBoard
- **Object Detection Models**: EfficientDet, Faster R-CNN, RetinaNet, YOLO

---

## 2. Pipeline Stages & Key Scripts

The project is divided into several key stages, each with corresponding scripts:

### Stage 1: Data Preprocessing & EDA

**Script**: `main_preprocess_pipeline.py`

This script now serves two main purposes, controlled by which function is called in `if __name__ == "__main__":`.

1.  **`data_preprocess()`**: Filters the raw COCO JSON dataset. Its primary role is to remove annotations and categories based on specific criteria (e.g., removing categories with few instances). A key fix was implemented to ensure that when annotations are removed, the corresponding entries in the `categories` list are also removed to prevent mismatches during training.

2.  **`data_ploting()`**: Performs a comprehensive Exploratory Data Analysis (EDA) on a single, preprocessed JSON file. It generates a suite of plots and detailed markdown reports in the `reports_V01_filnal/` directory to provide a deep understanding of the dataset's characteristics. The generated reports include:
    -   **Class Distribution**: Overall, and a focused view of the top 10 majority vs. bottom 10 minority classes.
    -   **Image Properties**: Distribution of image resolutions.
    -   **Bbox Properties**: Overall distribution of bbox area and aspect ratio.
    -   **Per-Class Bbox Analysis**: A detailed breakdown of bbox area and aspect ratio for each of the top classes.
    -   **Object Co-occurrence**: A heatmap showing which object classes frequently appear together.
    -   **Spatial Distribution**: A 2D histogram showing where objects are typically located within the images.
    -   **Objects per Image**: A histogram showing the number of objects per image.
    -   *Each report includes an interpretation guide explaining how to read the results and what they imply for model training.*

### Stage 2: Dataset Splitting

**Script**: `pid_preprocess/split_dataset.py`

-   Takes a single COCO JSON file and performs a stratified split into training, validation, and test sets. It supports multiple strategies (`random`, `hybrid`, `iterative`, etc.) to handle class imbalance during splitting.

### Stage 3: Model Training

**Script**: `pid_train.py`

-   The main entry point for training. It uses Hydra to manage all configurations.
-   The core training logic is defined in `pid_train/lightning_modules/object_detection_lit_module.py`.

### Stage 4: Model Evaluation

**Script**: `evaluate.py`

-   Evaluates a trained model checkpoint against a validation dataset.
-   It has been refactored to use the same tiling logic and dataset configuration as the training pipeline, ensuring consistency.
-   It performs evaluation on whole images by using a **sliding-window prediction** approach, stitching together predictions from multiple tiles.

### Stage 5: Verification & Visualization

**Script**: `visualization/visualize_tiling.py`

-   A command-line utility created to visually verify and debug the tiling process.
-   It takes a single image and tiling parameters as input and generates:
    1.  All individual tile images.
    2.  A side-by-side comparison image showing the original image and an overview with tile boundaries drawn on it.
    3.  A `tiling_visualization_guide.md` file explaining how to interpret the results to confirm success.

---

## 3. Key Architectural Concepts & Optimizations

Several important concepts and optimizations have been implemented:

### Centralized Tiling Logic
-   The core logic for generating tile coordinates is now centralized in the `generate_tiles` function within `utils/tile_utils.py`.
-   Both the training dataset (`BaseTiledDataset`) and the evaluation script (`evaluate.py`) use this shared utility, ensuring that tiling is performed identically in both stages.

### Optimized Dataset Creation
-   The `_create_tiles` method in `BaseTiledDataset`, which matches annotations to tiles, was identified as a performance bottleneck.
-   It has been refactored to use **NumPy vectorized operations**, replacing slow nested Python loops. This significantly speeds up the initial dataset loading and preprocessing time.

### Optimized Validation Loop
-   The validation metric calculation was identified as a major bottleneck, taking ~15 minutes per epoch due to the large number of classes and the computational cost of mAP.
-   The `ObjectDetectionLitModule` has been refactored to use the **"Accumulate then Compute"** pattern:
    1.  **`validation_step`**: Now only performs model inference and stores the predictions/targets in a list. It no longer performs any metric calculations.
    2.  **`on_validation_epoch_end`**: Gathers all stored predictions/targets and computes all metrics (F1, mAP, etc.) **once** at the end of the epoch.
-   **Selective mAP Calculation**: To further reduce validation time, the slow mAP calculation is now performed **only on the final validation epoch**. Faster metrics like F1-score are still calculated every epoch to monitor the general trend.
-   **Performance Monitoring**: The time taken for metric calculation is now logged as `val/metric_computation_time` for easy monitoring of the bottleneck.

### Configuration Management
-   The project heavily relies on Hydra for clean and modular configuration.
-   Model configurations (e.g., `efficientdet.yaml`) have been updated to inherit shared parameters like `num_classes` from `base.yaml` (e.g., `num_classes: ${num_classes}`), improving consistency.

### Debugging
-   The training script `pid_train.py` contained a `CUDA_LAUNCH_BLOCKING=1` flag for debugging. It has been identified that this severely degrades performance and should be **disabled** for normal training runs.