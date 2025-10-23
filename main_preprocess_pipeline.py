# main_preprocess_pipeline.py
import dataclasses
import os
from pathlib import Path

from pid_preprocess.data_loader import DataLoader, setting_categories_data, LoadedDaskData
from pid_preprocess.feature_engineering import add_bbox_features
from pid_preprocess import eda_runner # eda_runner에서 visualize_combined_class_distribution을 가져옴
from utils.file_utils import load_json_data


ver = 'V01'


def data_pipeline() -> None:
    print("data pipeline start")
    
    base_dir = Path(os.getcwd()).resolve()
    data_path = base_dir / "assets"
    
    save_path = base_dir / f"reports_{ver}"
    
    categories_df = setting_categories_data(data_path=data_path, json_file_name="categories.json")
    
    train_data_loader = DataLoader(dataPath=data_path, jsonDir="preprocessed_data_json", isLog=False)
    test_data_loader = DataLoader(dataPath=data_path, jsonDir="preprocessed_data_json", isLog=False)
    
    # train_data = train_data_loader.load_data("TL_prepro/TL_*_*/*.json")
    # test_data = test_data_loader.load_data("VL_prepro/VL_*_*/*.json")
    train_data = train_data_loader.load_data(f"TL_prepro/TL_{ver}_*/*.json")
    test_data = test_data_loader.load_data(f"VL_prepro/VL_{ver}_*/*.json")
    
    train_processed_annotations = add_bbox_features(train_data.annotations)
    train_data = dataclasses.replace(train_data, annotations=train_processed_annotations)
    test_precessed_annotations = add_bbox_features(test_data.annotations)
    test_data = dataclasses.replace(test_data, annotations=test_precessed_annotations)
    
    # eda_runner.run_bbox_analysis(train_data, categories_df, save_path, prefix="train")
    # eda_runner.run_bbox_analysis(test_data, categories_df, save_path, prefix="test")
    
    # eda_runner.run_class_distribution_analysis(train_data, categories_df, save_path, prefix="train")
    # eda_runner.run_class_distribution_analysis(test_data, categories_df, save_path, prefix="test")
    
    # eda_runner.run_image_property_analysis(train_data, save_path, prefix="train")
    # eda_runner.run_image_property_analysis(test_data, save_path, prefix="test")
    
    # TODO: 이 부분 수정이 필요한데, train하고 test를 합쳤을 때, 클래스가 총 몇개 나오는지도 반환해줘야함.
    eda_runner.run_train_test_comparison(train_data, test_data, categories_df, save_path)

    # TODO: train, test 합쳐진 상태에서의 클래스별 객체 수 분포 시각화하는 걸로 수정해야함. 지금은 따로따로 나오고 있음.
    eda_runner.visualize_combined_class_distribution(train_data, test_data, categories_df, save_path, prefix="combined_train_test")


if __name__ == "__main__":
    data_pipeline()
