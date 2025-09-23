# pid_preprocess/eda_runner.py
__all__ = [
    "run_bbox_analysis", "run_class_distribution_analysis", "run_image_property_analysis", "run_train_test_comparison"
]
from pathlib import Path

from pandas import DataFrame

from pid_preprocess.data_loader import LoadedDaskData
from visualization import plotting


def run_bbox_analysis(
    data: LoadedDaskData,
    categories_df: DataFrame,
    save_path: Path,
    prefix: str
):
    """
    바운딩 박스(BBox) 관련 모든 분석과 시각화를 실행합니다.
    (Area, Aspect Ratio, Spatial Distribution, W/H Scatter, Boxplots)
    """
    print(f"--- Starting BBox Analysis for '{prefix}' data ---")
    
    # BBox Area 분포 (로그 스케일)
    plotting.plot_dask_log_histogram(
        ddf=data.annotations,
        column='area',
        title=f'Bounding Box Area Distribution ({prefix.title()} Data)',
        xlabel='Area (log scale)',
        save_path=save_path / f'figures/{prefix}_area_distribution.png'
    )
    
    # BBox Aspect Ratio 분포
    plotting.plot_dask_histogram(
        ddf=data.annotations,
        column='aspect_ratio',
        title=f'Bounding Box Aspect Ratio Distribution ({prefix.title()} Data)',
        xlabel='Aspect Ratio (width / height)',
        show_kde=True,
        save_path=save_path / f'figures/{prefix}_aspect_ratio_hist_kde.png'
    )
    
    # BBox 공간 분포
    plotting.plot_dask_spatial_distribution(
        annotations_ddf=data.annotations,
        images_ddf=data.images,
        title=f'Spatial Distribution of BBox Centers ({prefix.title()} Data)',
        gridsize=150,
        save_path=save_path / f'figures/{prefix}_spatial_distribution.png'
    )
    
    # 주요 피처 분포 (w, h, area, aspect_ratio)
    features_to_plot = ['w', 'h', 'area', 'aspect_ratio']
    plotting.plot_dask_feature_distributions(
        ddf=data.annotations,
        features=features_to_plot,
        title=f'Bounding Box Feature Distributions ({prefix.title()} Data)',
        save_path=save_path / f'figures/{prefix}_feature_distributions.png'
    )

    # 너비/높이 산점도
    plotting.plot_dask_wh_scatter(
        ddf=data.annotations,
        title=f'Width vs Height Distribution ({prefix.title()} Data, 10% Sample)',
        sample_frac=0.1,
        save_path=save_path / f'figures/{prefix}_wh_scatter.png'
    )

    # 클래스별 BBox 면적 박스 플롯 (전체 카테고리)
    stats_df = plotting.calculate_dask_boxplot_stats(
        ddf=data.annotations,
        categories_df=categories_df,
        top_n=None # top_n=None으로 설정하여 전체 카테고리 대상
    )
    plotting.plot_boxplot_from_stats(
        stats_df=stats_df,
        title=f'Area Distribution by All Categories ({prefix.title()} Data)',
        xlabel='Area (log scale)',
        ylabel='Category Name',
        save_path=save_path / f'figures/{prefix}_category_area_boxplot_all.png'
    )


def run_class_distribution_analysis(
    data: LoadedDaskData,
    categories_df: DataFrame,
    save_path: Path,
    prefix: str
):
    """
    클래스 분포 관련 모든 분석과 시각화를 실행합니다.
    (전체 분포, 상위/하위 분포, 누적 분포)
    """
    print(f"--- Starting Class Distribution Analysis for '{prefix}' data ---")
    
    # 전체 클래스 분포
    plotting.plot_class_distribution(
        annotations_ddf=data.annotations,
        categories_df=categories_df,
        title=f'Class Distribution ({prefix.title()} Data, All Categories)',
        save_path=save_path / f'figures/{prefix}_class_distribution_all.png'
    )
    
    # 상위/하위 클래스 분포
    plotting.plot_top_bottom_classes(
        annotations_ddf=data.annotations,
        categories_df=categories_df,
        n_items=20,
        save_path=save_path / f'figures/{prefix}_top_bottom_classes.png'
    )
    
    # 누적 분포 곡선
    plotting.plot_cumulative_class_distribution(
        ddf=data.annotations,
        save_path=save_path / f'figures/{prefix}_cumulative_class_distribution.png'
    )


def run_image_property_analysis(
    data: LoadedDaskData,
    save_path: Path,
    prefix: str
):
    """
    이미지 자체의 속성(해상도, 이미지 당 객체 수)을 분석합니다.
    """
    print(f"--- Starting Image Property Analysis for '{prefix}' data ---")

    # --- 이미지별 객체 수 분석 ---
    object_counts_series = plotting.calculate_object_counts_per_image(
        ddf=data.annotations
    )
    plotting.summarize_object_counts(
        counts_series=object_counts_series,
        output_path=save_path / f"analysis/{prefix}_object_counts_summary.md"
    )
    plotting.plot_object_counts_distribution(
        counts_series=object_counts_series,
        title=f'Distribution of Objects per Image ({prefix.title()} Data)',
        save_path=save_path / f'figures/{prefix}_object_counts_dist.png'
    )

    # --- 이미지 해상도 분석 ---
    resolution_counts, stats, examples, max_img, min_img = plotting.analyze_image_resolutions(
        ddf=data.images
    )
    plotting.save_resolution_summary(
        resolution_counts=resolution_counts,
        resolution_stats=stats,
        image_id_examples=examples,
        max_area_image=max_img,
        min_area_image=min_img,
        output_path=save_path / f'analysis/{prefix}_resolution_summary.md'
    )
    plotting.plot_top_resolutions(
        resolution_counts_series=resolution_counts,
        title=f'Top 20 Most Common Image Resolutions ({prefix.title()} Data)',
        n_items=20,
        plot_top=True,
        save_path=save_path / f'figures/{prefix}_top_resolutions.png'
    )
    plotting.plot_top_resolutions(
        resolution_counts_series=resolution_counts,
        title=f'Bottom 20 Least Common Image Resolutions ({prefix.title()} Data)',
        n_items=20,
        plot_top=False,
        save_path=save_path / f'figures/{prefix}_bottom_resolutions.png'
    )


def run_train_test_comparison(
    train_data: LoadedDaskData,
    test_data: LoadedDaskData,
    categories_df: DataFrame,
    save_path: Path
):
    """
    Train 데이터셋과 Test 데이터셋을 비교 분석하여 리포트를 생성합니다.
    """
    print("--- Starting Train vs Test Comparison Analysis ---")
    
    train_imbalance_stats = plotting.analyze_class_imbalance(
        ddf=train_data.annotations
    )
    test_imbalance_stats = plotting.analyze_class_imbalance(
        ddf=test_data.annotations
    )
    
    non_overlapping_classes = plotting.find_non_overlapping_classes(
        train_ddf=train_data.annotations,
        test_ddf=test_data.annotations,
        categories_df=categories_df
    )
    
    plotting.save_combined_imbalance_report(
        train_stats=train_imbalance_stats,
        test_stats=test_imbalance_stats,
        non_overlapping_info=non_overlapping_classes,
        output_path=save_path / 'analysis/combined_class_imbalance_summary.md'
    )