# main_preprocess_pipeline.py

# 여기가 이제 pipeline의 출발지점!!

# 이 폴더는 EDA의 폴더니까 이것도 EDA만 하는 폴더임

# 일단 내가 했던 EDA를 전부 클래스로 만든 후, 저장하는 곳도 다 만들어 줘야 할 것 같음!
import dataclasses
import os
from pathlib import Path

from pid_preprocess.data_loader import DataLoader, setting_categories_data
from pid_preprocess.feature_engineering import add_bbox_features
from visualization import plotting


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
    
    
    # plotting.plot_dask_log_histogram(
    #     ddf=train_data.annotations,
    #     column='area',
    #     title='Bounding Box Area Distribution (Train Data)',
    #     xlabel='Area (log scale)',
    #     save_path=save_path / 'figures/train_area_distribution.png'
    # )
    
    # plotting.plot_dask_histogram(
    #     ddf=train_data.annotations,
    #     column='aspect_ratio',
    #     title='Bounding Box Aspect Ratio Distribution with KDE (Train Data)',
    #     xlabel='Aspect Ratio (width / height)',
    #     show_kde=True,
    #     save_path=save_path / 'figures/train_aspect_ratio_hist_kde.png'
    # )
    
    # plotting.plot_dask_log_histogram(
    #     ddf=test_data.annotations,
    #     column='area',
    #     title='Bounding Box Area Distribution (Train Data)',
    #     xlabel='Area (log scale)',
    #     save_path=save_path / 'figures/test_area_distribution.png'
    # )
    
    # plotting.plot_dask_histogram(
    #     ddf=test_data.annotations,
    #     column='aspect_ratio',
    #     title='Bounding Box Aspect Ratio Distribution with KDE (Train Data)',
    #     xlabel='Aspect Ratio (width / height)',
    #     show_kde=True,
    #     save_path=save_path / 'figures/test_aspect_ratio_hist_kde.png'
    # )
    
    # plotting.plot_dask_spatial_distribution(
    #     annotations_ddf=train_data.annotations,
    #     images_ddf=train_data.images,
    #     title='Spatial Distribution of Bounding Box Centers (Train Data)',
    #     gridsize=150, # 해상도를 더 높이고 싶다면 조절
    #     save_path=save_path / 'figures/train_spatial_distribution.png'
    # )
    
    # plotting.plot_dask_spatial_distribution(
    #     annotations_ddf=test_data.annotations,
    #     images_ddf=test_data.images,
    #     title='Spatial Distribution of Bounding Box Centers (Test Data)',
    #     gridsize=150, # 해상도를 더 높이고 싶다면 조절
    #     save_path=save_path / 'figures/test_spatial_distribution.png'
    # )
    
    
    # # 1. 기본 2x2 그리드로 4개 피처 그리기
    # features_to_plot = ['w', 'h', 'area', 'aspect_ratio']
    # plotting.plot_dask_feature_distributions(
    #     ddf=train_data.annotations,
    #     features=features_to_plot,
    #     title='Bounding Box Feature Distributions (Train Data)',
    #     save_path=save_path / 'figures/train_feature_distributions.png'
    # )

    # # 2. 3개의 열로 3개의 피처만 그려보기 (유연성 확인)
    # plotting.plot_dask_feature_distributions(
    #     ddf=train_data.annotations,
    #     features=['w', 'h', 'area'],
    #     ncols=3, # 열 개수를 3개로 지정
    #     title='Width, Height, and Area Distributions',
    #     save_path=save_path / 'figures/train_wha_distributions.png'
    # )
    
    # plotting.plot_dask_feature_distributions(
    #     ddf=test_data.annotations,
    #     features=features_to_plot,
    #     title='Bounding Box Feature Distributions (Test Data)',
    #     save_path=save_path / 'figures/test_feature_distributions.png'
    # )

    # plotting.plot_dask_feature_distributions(
    #     ddf=test_data.annotations,
    #     features=['w', 'h', 'area'],
    #     ncols=3, # 열 개수를 3개로 지정
    #     title='Width, Height, and Area Distributions',
    #     save_path=save_path / 'figures/test_wha_distributions.png'
    # )
    
    # # 1. Dask로 통계치 계산
    # train_stats_df = plotting.calculate_dask_boxplot_stats(
    #     ddf=train_data.annotations,
    #     categories_df=categories_df,
    #     top_n=20 # 상위 20개만 보기
    # )

    # # 2. 계산된 통계치로 플롯 그리기
    # plotting.plot_boxplot_from_stats(
    #     stats_df=train_stats_df,
    #     title='Top 20 Categories by Area Distribution',
    #     xlabel='Category Name',
    #     ylabel='Area (log scale)',
    #     save_path=save_path / 'figures/train_category_area_boxplot.png'
    # )
    
    # # 1. Dask로 통계치 계산
    # test_stats_df = plotting.calculate_dask_boxplot_stats(
    #     ddf=test_data.annotations,
    #     categories_df=categories_df,
    #     top_n=20 # 상위 20개만 보기
    # )

    # # 2. 계산된 통계치로 플롯 그리기
    # plotting.plot_boxplot_from_stats(
    #     stats_df=test_stats_df,
    #     title='Top 20 Categories by Area Distribution',
    #     xlabel='Category Name',
    #     ylabel='Area (log scale)',
    #     save_path=save_path / 'figures/test_category_area_boxplot.png'
    # )
    
    # # 1. Dask로 이미지별 객체 수 계산
    # train_object_counts_series = plotting.calculate_object_counts_per_image(
    #     ddf=train_data.annotations
    # )

    # # 2. 계산된 결과로 통계치 요약 출력
    # plotting.summarize_object_counts(
    #     counts_series=train_object_counts_series,
    #     output_path=save_path / "analysis/train_object_counts_summary.md"
    # )

    # # 3. 계산된 결과로 분포 시각화
    # plotting.plot_object_counts_distribution(
    #     counts_series=train_object_counts_series,
    #     title='Distribution of Objects per Image (Train Data)',
    #     save_path=save_path / 'figures/train_object_counts_dist.png'
    # )
    
    # test_object_counts_series = plotting.calculate_object_counts_per_image(
    #     ddf=test_data.annotations
    # )

    # plotting.summarize_object_counts(
    #     counts_series=test_object_counts_series,
    #     output_path=save_path / "analysis/test_object_counts_summary.md"
    # )

    # plotting.plot_object_counts_distribution(
    #     counts_series=test_object_counts_series,
    #     title='Distribution of Objects per Image (Test Data)',
    #     save_path=save_path / 'figures/test_object_counts_dist.png'
    # )

    # # 1. Dask로 해상도 분석 실행
    # train_resolution_counts, train_resolution_stats, train_resolution_examples, train_max_res_image, train_min_res_image = plotting.analyze_image_resolutions(
    #     ddf=train_data.images
    # )

    # # 2. 분석 결과 Markdown 리포트로 저장
    # plotting.save_resolution_summary(
    #     resolution_counts=train_resolution_counts,
    #     resolution_stats=train_resolution_stats,
    #     image_id_examples=train_resolution_examples,
    #     max_area_image=train_max_res_image,
    #     min_area_image=train_min_res_image,
    #     output_path=save_path / 'analysis/train_resolution_summary.md'
    # )

    # # 3. 분석 결과 시각화
    # plotting.plot_top_resolutions(
    #     resolution_counts_series=train_resolution_counts,
    #     title='Top 20 Most Common Image Resolutions (Train Data)',
    #     n_items=20,
    #     plot_top=True,
    #     save_path=save_path / 'figures/train_top_resolutions.png'
    # )
    
    # plotting.plot_top_resolutions(
    #     resolution_counts_series=train_resolution_counts,
    #     title='Bottom 20 Most Common Image Resolutions (Train Data)',
    #     n_items=20,
    #     plot_top=False,
    #     save_path=save_path / 'figures/train_bottom_resolutions.png'
    # )
    
    # test_resolution_counts, test_resolution_stats, test_resolution_examples, test_max_res_image, test_min_res_image = plotting.analyze_image_resolutions(
    #     ddf=test_data.images
    # )

    # plotting.save_resolution_summary(
    #     resolution_counts=test_resolution_counts,
    #     resolution_stats=test_resolution_stats,
    #     image_id_examples=test_resolution_examples,
    #     max_area_image=test_max_res_image,
    #     min_area_image=test_min_res_image,
    #     output_path=save_path / 'analysis/test_resolution_summary.md'
    # )

    # plotting.plot_top_resolutions(
    #     resolution_counts_series=test_resolution_counts,
    #     title='Top 20 Most Common Image Resolutions (Test Data)',
    #     n_items=20,
    #     save_path=save_path / 'figures/test_top_resolutions.png'
    # )
    
    # plotting.plot_top_resolutions(
    #     resolution_counts_series=test_resolution_counts,
    #     title='Bottom 20 Most Common Image Resolutions (Test Data)',
    #     n_items=20,
    #     save_path=save_path / 'figures/test_bottom_resolutions.png'
    # )
    
    # train_imbalance_stats = plotting.analyze_class_imbalance(
    #     ddf=train_data.annotations
    # )
    # test_imbalance_stats = plotting.analyze_class_imbalance(
    #     ddf=test_data.annotations
    # )
    
    # non_overlapping_classes = plotting.find_non_overlapping_classes(
    #     train_ddf=train_data.annotations,
    #     test_ddf=test_data.annotations,
    #     categories_df=categories_df
    # )
    
    # plotting.save_combined_imbalance_report(
    #     train_stats=train_imbalance_stats,
    #     test_stats=test_imbalance_stats,
    #     non_overlapping_info=non_overlapping_classes,
    #     output_path=save_path / 'analysis/combined_class_imbalance_summary.md'
    # )
    
    # plotting.plot_class_distribution(
    #     annotations_ddf=train_data.annotations,
    #     categories_df=categories_df,
    #     title='Class Distribution (All Categories)',
    #     save_path=save_path / 'figures/train_class_distribution_all.png'
    # )
    
    # plotting.plot_top_bottom_classes(
    #     annotations_ddf=train_data.annotations,
    #     categories_df=categories_df,
    #     n_items=20,
    #     save_path=save_path / 'figures/train_top_bottom_classes.png'
    # )
    
    # plotting.plot_cumulative_class_distribution(
    #     ddf=train_data.annotations,
    #     save_path=save_path / 'figures/train_cumulative_class_distribution.png'
    # )
    
    # plotting.plot_class_distribution(
    #     annotations_ddf=test_data.annotations,
    #     categories_df=categories_df,
    #     title='Class Distribution (All Categories)',
    #     save_path=save_path / 'figures/test_class_distribution_all.png'
    # )
    
    # plotting.plot_top_bottom_classes(
    #     annotations_ddf=test_data.annotations,
    #     categories_df=categories_df,
    #     n_items=20,
    #     save_path=save_path / 'figures/test_top_bottom_classes.png'
    # )
    
    # plotting.plot_cumulative_class_distribution(
    #     ddf=test_data.annotations,
    #     save_path=save_path / 'figures/test_cumulative_class_distribution.png'
    # )
    
    # 1. 클래스별 면적 박스 플롯
    train_stats_df = plotting.calculate_dask_boxplot_stats(
        ddf=train_data.annotations,
        categories_df=categories_df,
    )
    plotting.plot_boxplot_from_stats(
        stats_df=train_stats_df,
        title='Area Distribution by All Categories',
        xlabel='Area (log scale)',
        ylabel='Category Name',
        save_path=save_path / 'figures/train_category_area_boxplot.png'
    )

    # 2. 너비/높이 산점도
    plotting.plot_dask_wh_scatter(
        ddf=train_data.annotations,
        title='Width vs Height Distribution of All Objects (10% Sample)',
        sample_frac=0.1,
        save_path=save_path / 'figures/train_wh_scatter.png'
    )
    
    test_stats_df = plotting.calculate_dask_boxplot_stats(
        ddf=test_data.annotations,
        categories_df=categories_df,
    )
    plotting.plot_boxplot_from_stats(
        stats_df=test_stats_df,
        title='Area Distribution by All Categories',
        xlabel='Area (log scale)',
        ylabel='Category Name',
        save_path=save_path / 'figures/test_category_area_boxplot.png'
    )

    plotting.plot_dask_wh_scatter(
        ddf=train_data.annotations,
        title='Width vs Height Distribution of All Objects (10% Sample)',
        sample_frac=0.1,
        save_path=save_path / 'figures/test_wh_scatter.png'
    )
    
    

if __name__ == "__main__":
    data_pipeline()