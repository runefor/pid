# main_preprocess_pipeline.py
import dataclasses
import os
from pathlib import Path

from pid_preprocess.data_loader import DataLoader, setting_categories_data
from pid_preprocess.feature_engineering import add_bbox_features
from pid_preprocess import eda_runner


def data_pipeline():
    print("data pipeline start")
    
    base_dir = Path(os.getcwd()).resolve()
    data_path = base_dir / "assets"
    
    save_path = base_dir / "reports"
    
    categories_df = setting_categories_data(data_path=data_path, json_file_name="categories.json")
    
    train_data_loader = DataLoader(dataPath=data_path, jsonDir="preprocessed_data_json", isLog=False)
    test_data_loader = DataLoader(dataPath=data_path, jsonDir="preprocessed_data_json", isLog=False)
    
    train_data = train_data_loader.load_data("TL_prepro/TL_*_*/*.json")
    test_data = test_data_loader.load_data("VL_prepro/VL_*_*/*.json")
    
    train_processed_annotations = add_bbox_features(train_data.annotations) 
    train_data = dataclasses.replace(train_data, annotations=train_processed_annotations)
    test_precessed_annotations = add_bbox_features(test_data.annotations)
    test_data = dataclasses.replace(test_data, annotations=test_precessed_annotations)
    
    # eda_runner.run_bbox_analysis(train_data, categories_df, save_path, prefix="train")
    # eda_runner.run_bbox_analysis(test_data, categories_df, save_path, prefix="test")
    
    # eda_runner.run_class_distribution_analysis(train_data, categories_df, save_path, prefix="train")
    # eda_runner.run_class_distribution_analysis(test_data, categories_df, save_path, prefix="test")
    
    eda_runner.run_image_property_analysis(train_data, save_path, prefix="train")
    eda_runner.run_image_property_analysis(test_data, save_path, prefix="test")
    
    eda_runner.run_train_test_comparison(train_data, test_data, categories_df, save_path)


if __name__ == "__main__":
    data_pipeline()