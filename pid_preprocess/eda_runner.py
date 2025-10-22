# pid_preprocess/eda_runner.py
__all__ = [
    "run_bbox_analysis", "run_class_distribution_analysis", "run_image_property_analysis", "run_train_test_comparison",
    "visualize_combined_class_distribution"
]
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame
import pandas as pd

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


def visualize_combined_class_distribution(
    train_data: LoadedDaskData, # Assuming DataLoader returns LoadedDaskData
    test_data: LoadedDaskData, # Assuming DataLoader returns LoadedDaskData
    categories_df: pd.DataFrame,
    save_path: Path,
    prefix: str = "combined"
) -> None:
    """
    Train 및 Test 데이터셋을 합친 상태에서의 클래스별 객체 수 분포를 시각화합니다.
    """
    print(f"📊 Starting combined class distribution visualization for '{prefix}' data...")

    figure_path = save_path / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)

    # Combine annotations from train and test
    # Ensure dask.dataframe is imported for dd.concat
    import dask.dataframe as dd
    combined_annotations_ddf = dd.concat([train_data.annotations, test_data.annotations], ignore_index=True)

    # Compute class counts
    class_counts_series = combined_annotations_ddf['category_id'].value_counts().compute()
    class_counts = Counter(class_counts_series.to_dict())

    # Map category IDs to names
    id_to_name = categories_df['name'].to_dict()

    # Sort class distribution by count
    class_distribution = sorted(
        [(id_to_name.get(cat_id, f"Unknown({cat_id})"), count) for cat_id, count in class_counts.items()],
        key=lambda item: item[1],
        reverse=True
    )

    if not class_distribution:
        print("   ⚠️ No annotations found in the combined dataset.")
        return

    # Visualization
    _plotting_class_distribution(class_distribution, f'Class Distribution in Combined Train and Test Data ({prefix.title()})', figure_path / f"{prefix}_class_distribution.png")


def _plotting_class_distribution(class_distribution: list, title: str, save_file: Path) -> None:
    """
    클래스 분포를 막대 그래프로 시각화하고 저장합니다.
    (내부 사용을 위해 _ 접두사 추가)
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, max(10, len(class_distribution) * 0.4))) # 클래스 수에 따라 높이 조절

    class_names, counts = zip(*class_distribution)
    sns.barplot(x=list(counts), y=list(class_names), orient='h', ax=ax)

    ax.bar_label(ax.containers[0], fmt='{:,.0f}', padding=5, fontsize=10)

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Annotations', fontsize=12)
    ax.set_ylabel('Class Name', fontsize=12)
    ax.set_xlim(right=max(counts) * 1.15) # 숫자 레이블이 잘리지 않도록 x축 범위 확장
    plt.tight_layout()
    plt.savefig(save_file)
    plt.close(fig) # 메모리 해제를 위해 figure 닫기
    print(f"   ✅ Visualization saved to: {save_file}")