from pathlib import Path
from typing import Optional

import dask
import dask.array as da
import dask.dataframe as dd
from dask import compute
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd


plt.style.use('seaborn-v0_8-whitegrid') 

def plot_dask_log_histogram(
    ddf: dd.DataFrame, 
    column: str, 
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    bins: int = 50,
    save_path: Optional[Path] = None
):
    """
    Dask 데이터프레임의 특정 컬럼에 대한 로그 스케일 히스토그램을 그립니다.

    Args:
        ddf (dd.DataFrame): 대상 Dask 데이터프레임.
        column (str): 히스토그램을 그릴 컬럼명.
        title (str): 플롯의 제목.
        xlabel (str): x축 라벨.
        ylabel (str, optional): y축 라벨. Defaults to "Frequency".
        bins (int, optional): 히스토그램의 빈 개수. Defaults to 50.
        save_path (Path, optional): None이 아니면 플롯을 해당 경로에 저장. Defaults to None.
        
    Returns:
        tuple: (matplotlib.figure, matplotlib.axes) 객체를 반환.
    """
    # 1. 히스토그램 계산 계획 세우기
    col_min, col_max = dask.compute(ddf[column].min(), ddf[column].max())
    
    # 최소값이 0이하일 경우 로그 스케일 오류 방지
    start_val = max(col_min, 1e-6) 
    bin_edges_plan = np.logspace(np.log10(start_val), np.log10(col_max), bins)

    counts_plan, bin_edges_plan = da.histogram(ddf[column], bins=bin_edges_plan)

    # 2. 실제 계산 실행 (결과는 작은 numpy 배열)
    counts, bin_edges = dask.compute(counts_plan, bin_edges_plan)

    # 3. Matplotlib로 결과 플로팅
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bin_widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], counts, width=bin_widths, align='edge', alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig) # 파일을 저장한 후에는 창을 닫아줍니다.
    else:
        plt.show()
        
    return fig, ax


def plot_dask_histogram(
    ddf: dd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    bins: int = 50,
    show_kde: bool = False,
    save_path: Optional[Path] = None
) -> tuple:
    """
    Dask 데이터프레임의 특정 컬럼에 대한 일반 히스토그램을 그립니다.
    선택적으로 KDE(밀도 곡선)를 함께 표시할 수 있습니다.
    """
    print(f"'{column}' 컬럼에 대한 히스토그램 계산을 시작합니다...")
    # 1. 히스토그램 계산 계획 세우기
    col_min, col_max = dask.compute(ddf[column].min(), ddf[column].max())
    
    # linspace는 구간 경계를 생성하므로, 구간(bin) 개수보다 1 많게 설정
    bin_edges_plan = np.linspace(col_min, col_max, bins + 1)
    
    counts_plan, bin_edges_plan = da.histogram(ddf[column], bins=bin_edges_plan)

    # 2. 실제 계산 실행
    counts, bin_edges = dask.compute(counts_plan, bin_edges_plan)
    print("계산 완료.")

    # 3. Matplotlib로 결과 플로팅
    print("시각화 생성 중...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bin_widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], counts, width=bin_widths, align='edge', alpha=0.7, label='Frequency')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", ls="--", linewidth=0.5)

    # 4. KDE 곡선 그리기 (show_kde=True일 경우)
    if show_kde:
        # KDE는 별도의 y축(오른쪽)을 사용하도록 설정
        ax2 = ax.twinx()
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        sns.kdeplot(x=bin_centers, weights=counts, color='red', ax=ax2, label='Density')
        ax2.set_ylabel('Density', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 범례(legend)를 합쳐서 표시
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
    return fig, ax


def plot_dask_spatial_distribution(
    annotations_ddf: dd.DataFrame,
    images_ddf: dd.DataFrame,
    title: str,
    gridsize: int = 100,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    annotations와 images Dask 데이터프레임을 병합하여,
    정규화된 바운딩 박스 중심점의 공간적 분포를 2D 히스토그램으로 그립니다.
    """
    print("바운딩 박스 공간 분포 시각화를 시작합니다...")
    
    # --- 1. 데이터 준비 (병합 및 정규화) ---
    print("메모리 효율적인 병합을 위해 images 데이터를 로드합니다...")
    # 작은 images_ddf만 Pandas로 변환하여 메모리에 로드
    images_pdf = images_ddf[['id', 'width', 'height']].rename(columns={'id': 'image_id'}).compute()
    print("로드 완료.")

    # annotations의 각 파티션에 images_pdf를 병합하는 함수
    def merge_and_normalize(partition_df, lookup_df):
        merged = pd.merge(partition_df, lookup_df, on='image_id')
        merged['norm_center_x'] = merged['center_x'] / merged['width']
        merged['norm_center_y'] = merged['center_y'] / merged['height']
        return merged[['norm_center_x', 'norm_center_y']]

    # Dask에게 결과물의 스키마(meta)를 알려줌
    meta_df = pd.DataFrame({
        'norm_center_x': pd.Series(dtype='float64'),
        'norm_center_y': pd.Series(dtype='float64'),
    })
    
    # map_partitions으로 병합 및 정규화 계획 수립
    normalized_coords_ddf = annotations_ddf[['image_id', 'center_x', 'center_y']].map_partitions(
        merge_and_normalize,
        lookup_df=images_pdf,
        meta=meta_df
    )

    # --- 2. 2D 히스토그램 계산 ---
    print("2D 히스토그램 계산 중 (연산 트리거)...")
    bins = [gridsize, gridsize]
    range_ = [[0, 1], [0, 1]]

    H, xedges, yedges = da.histogram2d(
        normalized_coords_ddf['norm_center_x'].values,
        normalized_coords_ddf['norm_center_y'].values,
        bins=bins,
        range=range_
    )
    computed_H, computed_xedges, computed_yedges = dask.compute(H, xedges, yedges)
    print("계산 완료.")

    # --- 3. Matplotlib으로 시각화 ---
    print("Matplotlib으로 시각화 생성 중...")
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(4, 4, figure=fig)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    fig.suptitle(title, y=0.93, fontsize=16)

    # 메인 2D 히스토그램
    im = ax_main.pcolormesh(computed_xedges, computed_yedges, computed_H.T, cmap='viridis')
    ax_main.set_xlabel('Normalized Center X')
    ax_main.set_ylabel('Normalized Center Y')

    # 상단 1D 히스토그램
    x_counts = computed_H.sum(axis=1)
    ax_top.bar(computed_xedges[:-1], x_counts, width=np.diff(computed_xedges), align='edge')
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_yticks([])

    # 오른쪽 1D 히스토그램
    y_counts = computed_H.sum(axis=0)
    ax_right.barh(computed_yedges[:-1], y_counts, height=np.diff(computed_yedges), align='edge')
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xticks([])

    # y축 뒤집기 (이미지 좌표계와 동일하게)
    ax_main.invert_yaxis()
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    gs.update(wspace=0.05, hspace=0.05)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
    return fig


def plot_dask_feature_distributions(
    ddf: dd.DataFrame,
    features: list[str],
    title: str,
    ncols: int = 2,
    bins: int = 100,
    aspect_ratio_range: Optional[tuple[float, float]] = (0, 5),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Dask 데이터프레임의 여러 숫자 컬럼(피처)에 대한 분포를
    하나의 그리드에 히스토그램으로 그립니다.

    Args:
        ddf (dd.DataFrame): 대상 Dask 데이터프레임.
        features (list[str]): 히스토그램을 그릴 컬럼명 리스트.
        title (str): 전체 플롯의 제목.
        ncols (int, optional): 그리드의 열 개수. Defaults to 2.
        bins (int, optional): 각 히스토그램의 빈 개수. Defaults to 100.
        aspect_ratio_range (tuple, optional): 'aspect_ratio' 컬럼의 범위를 제한. None이면 제한 없음.
        save_path (Optional[Path], optional): None이 아니면 플롯을 해당 경로에 저장.
    """
    print("여러 피처에 대한 히스토그램 계산 및 시각화를 시작합니다...")
    
    # 1. 서브플롯 그리드 생성
    nrows = (len(features) + ncols - 1) // ncols  # 필요한 행 개수 자동 계산
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 5))
    
    # axes가 1차원 배열이 되도록 flatten() 처리 (plot이 1개일 때도 에러 방지)
    axes = np.array(axes).flatten()

    # 2. 각 피처에 대해 반복하며 히스토그램 계산 및 그리기
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # NaN/inf 값 제외
        valid_data = ddf[feature].dropna()
        valid_data = valid_data[~valid_data.isin([np.inf, -np.inf])]

        min_val, max_val = dask.compute(valid_data.min(), valid_data.max())
        
        # 피처에 따른 조건부 bin 생성
        plot_bins = bins
        if feature == 'area' and min_val > 0:
            plot_bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)
            ax.set_xscale('log')
        else:
            if feature == 'aspect_ratio' and aspect_ratio_range:
                min_val = max(aspect_ratio_range[0], min_val)
                max_val = min(aspect_ratio_range[1], max_val)
            plot_bins = np.linspace(min_val, max_val, bins)

        # Dask로 히스토그램 계산
        counts, bin_edges = da.histogram(valid_data.values, bins=plot_bins)
        computed_counts, computed_bin_edges = dask.compute(counts, bin_edges)
        
        # 서브플롯에 막대그래프 그리기
        ax.bar(
            computed_bin_edges[:-1], 
            computed_counts, 
            width=np.diff(computed_bin_edges), 
            align='edge',
            alpha=0.8
        )
        ax.set_title(f'Distribution of {feature.capitalize()}', fontsize=14)
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.grid(True, ls="--", linewidth=0.5)

    # 남는 빈 서브플롯은 보이지 않게 처리
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    fig.suptitle(title, fontsize=20, y=1.02)
    fig.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def calculate_dask_boxplot_stats( # TODO: 이건 시각화 코드가 아니긴 해서 분리가 필요할 듯
    ddf: dd.DataFrame,
    categories_df: pd.DataFrame,
    group_col: str = 'category_id',
    value_col: str = 'area',
    top_n: Optional[int] = 15
) -> pd.DataFrame:
    """
    Dask 데이터프레임에서 박스 플롯을 그리기 위한 통계치를 계산합니다.
    
    Args:
        ddf (dd.DataFrame): annotations Dask 데이터프레임.
        categories_df (pd.DataFrame): 카테고리 이름 정보가 있는 Pandas 데이터프레임.
        group_col (str): 그룹화할 컬럼명 (예: 'category_id').
        value_col (str): 통계치를 계산할 컬럼명 (예: 'area').
        top_n (int, optional): 상위 n개 카테고리만 선택. None이면 전체.
    
    Returns:
        pd.DataFrame: 박스 플롯 통계치가 계산된 Pandas 데이터프레임.
    """
    print(f"'{value_col}'에 대한 박스 플롯 통계치 계산을 시작합니다...")
    
    # 1. Top N 카테고리 필터링
    if top_n:
        top_categories = ddf[group_col].value_counts().nlargest(top_n).index.compute()
        filtered_ddf = ddf[ddf[group_col].isin(top_categories)]
    else:
        filtered_ddf = ddf

    # 2. Dask로 Quantile 계산 계획 수립
    stats_series_ddf = filtered_ddf.groupby(group_col)[value_col].apply(
        lambda x: x.quantile([0.25, 0.50, 0.75]),
        meta=pd.Series(dtype='float64', name=value_col)
    )

    # 3. 실제 계산 실행 및 Pandas로 변환
    stats_series_pdf = stats_series_ddf.compute()
    stats_pdf = stats_series_pdf.unstack().rename(columns={0.25: 'q1', 0.50: 'med', 0.75: 'q3'})

    # 4. Whisker 계산
    stats_pdf = pd.merge(stats_pdf, categories_df, left_index=True, right_on='id')
    iqr = stats_pdf['q3'] - stats_pdf['q1']
    stats_pdf['whislo'] = (stats_pdf['q1'] - 1.5 * iqr).clip(lower=0)
    stats_pdf['whishi'] = stats_pdf['q3'] + 1.5 * iqr
    
    print("통계치 계산 완료.")
    return stats_pdf


def plot_boxplot_from_stats(
    stats_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    ylog: bool = True,
    save_path: Optional[Path] = None
) -> tuple:
    """
    미리 계산된 통계치 데이터프레임으로 박스 플롯을 그립니다.
    """
    print("박스 플롯 시각화 생성 중...")
    # 1. Matplotlib의 bxp 함수에 맞는 형식으로 데이터 변환
    plot_stats = []
    # 중앙값(med) 기준으로 정렬하여 보기 좋게 만듬
    sorted_df = stats_df.sort_values('med', ascending=False)
    
    for _, row in sorted_df.iterrows():
        plot_stats.append({
            'label': row['name'], # 카테고리 이름 사용
            'med': row['med'], 'q1': row['q1'], 'q3': row['q3'],
            'whislo': row['whislo'], 'whishi': row['whishi'],
            'fliers': []  # 이상치는 따로 계산하지 않았으므로 비워둠
        })

    # 2. 시각화
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bxp(plot_stats, showfliers=False, vert=True, patch_artist=True)
    
    if ylog:
        ax.set_yscale('log')
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
    return fig, ax

def calculate_object_counts_per_image( # TODO: 이것도 계산하는 부분인데 다른쪽에다가 넣어두면 좋을 것 같긴함.
    ddf: dd.DataFrame, 
    group_col: str = 'image_id'
) -> pd.Series:
    """
    Dask 데이터프레임에서 이미지별 객체 수를 계산하여 Pandas Series로 반환합니다.
    """
    print("이미지별 객체 수 계산을 시작합니다 (Dask 연산)...")
    object_counts = ddf[group_col].value_counts()
    object_counts_pd = object_counts.compute()
    print("계산 완료.")
    return object_counts_pd

def summarize_object_counts(
    counts_series: pd.Series,
    output_path: Path, # 👈 print 대신 파일 경로를 받도록 변경
    find_single_objects: bool = True
):
    """
    이미지별 객체 수(Pandas Series)를 받아 통계치를 Markdown 리포트로 저장합니다.
    """
    print(f"'{output_path}' 경로에 분석 요약 리포트를 저장합니다...")
    
    # 1. Markdown 내용을 담을 리스트를 생성합니다.
    report_lines = []
    
    # 2. 각 분석 결과를 Markdown 형식의 문자열로 추가합니다.
    report_lines.append("# 이미지별 객체 수 분석 리포트\n\n")
    
    report_lines.append("## 통계 요약\n")
    report_lines.append(f"- **고유 이미지 수**: {len(counts_series)}\n")
    report_lines.append(f"- **이미지 당 평균 객체 수**: {counts_series.mean():.2f}\n")
    report_lines.append(f"- **가장 객체가 많은 이미지의 객체 수**: {counts_series.max()}\n")
    report_lines.append(f"- **가장 객체가 적은 이미지의 객체 수**: {counts_series.min()}\n\n")
    
    max_objects_id = counts_series.idxmax()
    report_lines.append("## 주요 이미지 정보\n")
    report_lines.append(f"- **가장 객체가 많은 이미지 ID**: `{max_objects_id}` (객체 수: {counts_series.max()})\n\n")
    max_objects_ids = counts_series.nlargest(5)
    report_lines.append("### 가장 객체가 많은 이미지들 Top 5\n")
    report_lines.append(max_objects_ids.to_frame(name="Object Count").to_markdown())
    report_lines.append("\n\n")

    min_objects_ids = counts_series.nsmallest(5)
    report_lines.append("### 가장 객체가 적은 이미지들 Top 5\n")
    
    report_lines.append(min_objects_ids.to_frame(name="Object Count").to_markdown())
    report_lines.append("\n\n")
    
    if find_single_objects:
        single_object_images = counts_series[counts_series == 1]
        report_lines.append(f"## 객체가 1개인 이미지 분석\n")
        report_lines.append(f"- **총 개수**: {len(single_object_images)}\n\n")
        report_lines.append("### ID 목록 (상위 100개)\n")
        
        report_lines.append(single_object_images.head(100).to_frame(name="Object Count").to_markdown())
        report_lines.append("\n")

    # 3. 리스트를 하나의 문자열로 합치고 파일에 씁니다.
    report_content = "".join(report_lines)
    
    # 저장 경로의 부모 폴더가 없으면 자동으로 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("리포트 저장을 완료했습니다.")


def plot_object_counts_distribution(
    counts_series: pd.Series,
    title: str,
    xlabel: str = "Number of Objects per Image",
    ylabel: str = "Number of Images",
    bins: int = 100,
    save_path: Optional[Path] = None
) -> tuple:
    """
    이미지별 객체 수 분포를 히스토그램으로 그립니다.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts_series.hist(ax=ax, bins=bins)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(False)
    
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
    return fig, ax


def analyze_image_resolutions(ddf: dd.DataFrame, n_examples: int = 3) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    images Dask 데이터프레임에서 해상도 분석을 수행합니다. (수정된 버전)
    """
    print("이미지 해상도 분석을 시작합니다 (Dask 연산)...")
    
    if 'area' not in ddf.columns:
        ddf['area'] = ddf['width'] * ddf['height']
    if 'resolution' not in ddf.columns:
        ddf['resolution'] = ddf['width'].astype(str) + 'x' + ddf['height'].astype(str)
    
    # 1. 최고/최저 해상도 이미지 정보를 먼저 확정합니다.
    print("최고/최저 해상도 이미지를 찾는 중...")
    # 1-1. 인덱스 값을 먼저 계산
    max_area_image_ddf = ddf.nlargest(1, 'area')
    min_area_image_ddf = ddf.nsmallest(1, 'area')
    

    # 2. 나머지 통계치들을 계산합니다.
    print("나머지 통계치들을 계산 중...")
    resolution_counts = ddf['resolution'].value_counts()
    resolution_stats = ddf[['width', 'height']].describe()
    resolution_id_examples = ddf.groupby('resolution')['id'].apply(
        lambda s: s.head(n_examples).tolist(), meta=('id', 'object')
    )
    
    # 나머지 연산들만 한 번에 실행
    counts_pd, stats_pd, examples_pd, max_area_image_pd, min_area_image_pd = compute(
        resolution_counts, resolution_stats, resolution_id_examples,
        max_area_image_ddf, min_area_image_ddf
    )
    
    counts_pd = counts_pd.sort_values(ascending=False)
    print("계산 완료.")
    
    return counts_pd, stats_pd, examples_pd, max_area_image_pd, min_area_image_pd

def save_resolution_summary(
    resolution_counts: pd.Series,
    resolution_stats: pd.DataFrame,
    image_id_examples: pd.Series,
    max_area_image: pd.Series,
    min_area_image: pd.Series,
    output_path: Path,
    top_n: int = 20,
    bottom_n: int = 20,
):
    """
    해상도 분석 결과를 Markdown 리포트로 저장합니다.
    """
    print(f"'{output_path}' 경로에 해상도 분석 리포트를 저장합니다...")
    
    # 1. 모든 데이터를 하나의 DataFrame으로 먼저 합칩니다.
    counts_df = resolution_counts.to_frame(name="Image Count")
    examples_df = image_id_examples.to_frame(name="Example Image IDs")
    
    # join을 사용하여 두 정보를 합칩니다.
    combined_df = counts_df.join(examples_df)
    
    # 2. 합쳐진 DataFrame을 'Image Count' 기준으로 확실하게 정렬합니다.
    # 이 단계가 모든 정렬 문제를 해결합니다.
    sorted_df = combined_df.sort_values("Image Count", ascending=False)
    
    # --- 3. 이제 정렬된 DataFrame에서 상위/하위 데이터를 추출합니다. ---
    report_lines = ["# 이미지 해상도 분석 리포트\n\n"]
    
    max_img_row = max_area_image.iloc[0]
    min_img_row = min_area_image.iloc[0]
    
    report_lines.append("## 주요 통계\n")
    report_lines.append(f"- **가장 높은 해상도 (면적 기준)**: "
                        f"`{max_img_row['width']}x{max_img_row['height']}` "
                        f"(ID: `{max_img_row['id']}`)\n")
    report_lines.append(f"- **가장 낮은 해상도 (면적 기준)**: "
                        f"`{min_img_row['width']}x{min_img_row['height']}` "
                        f"(ID: `{min_img_row['id']}`)\n\n")
    
    report_lines.append("## 너비 및 높이 통계\n")
    report_lines.append(resolution_stats.to_markdown())
    report_lines.append("\n\n")
    
    report_lines.append(f"## 가장 많이 보이는 {top_n}개 해상도\n")
    report_lines.append(sorted_df.head(top_n).to_markdown())
    report_lines.append("\n\n")
    
    report_lines.append(f"## 가장 드문 {bottom_n}개 해상도\n")
    # 하위 데이터는 오름차순으로 보여주는 것이 더 자연스러울 수 있습니다.
    report_lines.append(sorted_df.tail(bottom_n).sort_values("Image Count", ascending=True).to_markdown())
    report_lines.append("\n\n")
    
    report_content = "".join(report_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("리포트 저장을 완료했습니다.")
    

def plot_top_resolutions(
    resolution_counts_series: pd.Series,
    title: str,
    n_items: int = 20,
    plot_top: bool = True,
    save_path: Optional[Path] = None,
) -> tuple:
    """
    상위 n개 해상도 분포를 수평 막대그래프로 그립니다.
    """
    if plot_top:
        data_to_plot = resolution_counts_series.head(n_items)
    else:
        # 하위 n개는 오름차순으로 정렬해야 보기 좋습니다.
        data_to_plot = resolution_counts_series.tail(n_items).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.barplot(x=data_to_plot.values, y=data_to_plot.index, orient='h', palette='viridis', ax=ax)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_ylabel('Resolution (Width x Height)', fontsize=12)
    
    for index, value in enumerate(data_to_plot):
        ax.text(value, index, f' {value}', va='center')

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
    return fig, ax


def analyze_class_imbalance(ddf: dd.DataFrame) -> dict: # TODO: 이 함수도 계산하는 부분인데 다른쪽에다가 넣어두면 좋을 것 같긴함.
    """
    클래스 개수 시리즈를 받아 불균형 관련 통계치를 계산합니다.
    """
    print("클래스 불균형 통계치를 계산합니다...")
    
    # 1. Dask로 클래스별 개수 계산 (계획)
    class_counts: pd.DataFrame = ddf['category_id'].value_counts().compute()
    
    # 2. 필요한 모든 통계치 계산을 Dask 연산으로 계획
    tasks = {
        'total_classes': class_counts.shape[0],
        'min_count': class_counts.min(),
        'max_count': class_counts.max(),
        'mean_count': class_counts.mean(),
        'median_count': class_counts.median(),
    }
    
    # 3. 모든 계획을 한 번에 실행
    results = dask.compute(tasks)[0]
    
    # 4. 계산된 결과를 바탕으로 추가 정보 생성
    results['max_min_ratio'] = results['max_count'] / results['min_count'] if results['min_count'] > 0 else float('inf')
    
    # 5. 누적 분포 계산 (이 부분은 계산된 Pandas Series로 수행하는 것이 더 효율적)
    cumulative_percentage = class_counts.cumsum() / class_counts.sum()
    results['classes_for_50_percent'] = (cumulative_percentage < 0.50).sum() + 1
    results['classes_for_80_percent'] = (cumulative_percentage < 0.80).sum() + 1
    results['classes_for_95_percent'] = (cumulative_percentage < 0.95).sum() + 1
    
    print("계산 완료.")
    return results

def find_non_overlapping_classes(
    train_ddf: dd.DataFrame,
    test_ddf: dd.DataFrame,
    categories_df: pd.DataFrame
) -> dict:
    """
    Train과 Test 데이터셋 간에 서로 겹치지 않는 클래스를 찾아서 반환합니다.
    """
    print("Train/Test 데이터셋 간 클래스 ID 목록을 비교합니다...")
    
    # 1. 각 데이터셋의 고유 카테고리 ID 목록을 계산합니다.
    train_ids_task = train_ddf['category_id'].unique()
    test_ids_task = test_ddf['category_id'].unique()
    
    train_ids_pd, test_ids_pd = compute(train_ids_task, test_ids_task)
    
    # 2. set(집합)으로 변환하여 차집합을 구합니다.
    train_id_set = set(train_ids_pd)
    test_id_set = set(test_ids_pd)
    
    train_only_ids = train_id_set - test_id_set
    test_only_ids = test_id_set - train_id_set
    
    # 3. ID를 실제 클래스 이름으로 변환합니다.
    category_map = pd.Series(categories_df['name'].values, index=categories_df.index).to_dict()
    
    
    train_only_names = [category_map.get(id, f"Unknown ID: {id}") for id in train_only_ids]
    test_only_names = [category_map.get(id, f"Unknown ID: {id}") for id in test_only_ids]
    
    print("비교 완료.")
    
    return {
        'train_only': sorted(train_only_names),
        'test_only': sorted(test_only_names)
    }

def save_combined_imbalance_report(
    train_stats: dict,
    test_stats: dict,
    non_overlapping_info: dict,
    output_path: Path
):
    """
    Train/Test 클래스 불균형 통계치와 겹치지 않는 클래스 정보를
    하나의 종합 Markdown 리포트로 저장합니다.
    """
    print(f"'{output_path}' 경로에 종합 클래스 불균형 리포트를 저장합니다...")

    # 읽기 쉬운 라벨 매핑
    key_labels = {
        'total_classes': '총 클래스 수',
        'min_count': '최소 객체 수',
        'max_count': '최대 객체 수',
        'mean_count': '평균 객체 수',
        'median_count': '객체 수 중앙값',
        'max_min_ratio': '최대/최소 클래스 비율',
        'classes_for_50_percent': '데이터의 50%를 차지하는 상위 클래스 수',
        'classes_for_80_percent': '데이터의 80%를 차지하는 상위 클래스 수',
        'classes_for_95_percent': '데이터의 95%를 차지하는 상위 클래스 수',
    }
    
    # --- 리포트 내용 생성 ---
    report_lines = ["# Train vs Test 클래스 불균형 종합 분석 리포트\n\n"]
    
    # Train 데이터 통계치 섹션
    report_lines.append("## Train 데이터셋 불균형 지표\n")
    for key, value in train_stats.items():
        label = key_labels.get(key, key)
        formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        report_lines.append(f"- **{label}**: {formatted_value}\n")
    report_lines.append("\n")

    # Test 데이터 통계치 섹션
    report_lines.append("## Test 데이터셋 불균형 지표\n")
    for key, value in test_stats.items():
        label = key_labels.get(key, key)
        formatted_value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
        report_lines.append(f"- **{label}**: {formatted_value}\n")
    report_lines.append("\n")
    
    # 겹치지 않는 클래스 분석 섹션
    report_lines.append("## 데이터셋 간 클래스 비교\n")
    train_only = non_overlapping_info['train_only']
    test_only = non_overlapping_info['test_only']
    
    report_lines.append(f"- **Train 데이터셋에만 존재하는 클래스**: {len(train_only)}개\n")
    if train_only:
        report_lines.append("  - " + ", ".join(f"`{name}`" for name in train_only) + "\n")
        
    report_lines.append(f"- **Test 데이터셋에만 존재하는 클래스**: {len(test_only)}개\n")
    if test_only:
        report_lines.append("  - " + ", ".join(f"`{name}`" for name in test_only) + "\n")

    # --- 파일 저장 ---
    report_content = "".join(report_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("리포트 저장을 완료했습니다.")


def plot_class_distribution(
    annotations_ddf: dd.DataFrame,
    categories_df:pd.DataFrame,
    title: str,
    top_n: Optional[int] = None,
    show_counts: bool = True,
    x_log_scale: bool = False,
    save_path: Optional[Path] = None
) -> tuple:
    """
    클래스별 어노테이션 개수 분포를 수평 막대그래프로 그립니다.

    Args:
        annotations_df (dd.DataFrame): annotations 데이터프레임.
        categories_df (pd.DataFrame): 카테고리 정보가 있는 Pandas 데이터프레임.
        title (str): 플롯의 제목.
        top_n (int, optional): 상위 n개 클래스만 표시. None이면 전체 표시.
        show_counts (bool, optional): 막대 옆에 개수를 표시할지 여부.
        save_path (Optional[Path], optional): None이 아니면 플롯을 해당 경로에 저장.
    """
    print("클래스 분포 시각화를 생성합니다...")
    # 1. Dask로 클래스별 개수 계산 (계획)
    class_counts_ddf = annotations_ddf['category_id'].value_counts()

    # 2. 상위 N개만 선택 (계획)
    if top_n:
        class_counts_ddf = class_counts_ddf.nlargest(n=top_n)
    
    # 3. Dask Merge를 사용해 카테고리 이름 결합 (계획)
    # value_counts 결과(Series)를 merge를 위해 DataFrame으로 변환
    counts_to_merge_ddf = class_counts_ddf.to_frame(name='count')
    # categories_df의 'id'와 counts_ddf의 인덱스(category_id)를 기준으로 merge
    merged_ddf = counts_to_merge_ddf.merge(
        categories_df,
        left_index=True,
        right_index=True
    )

    # 4. 실제 계산 실행 (시각화에 필요한 작은 데이터만 Pandas로 변환)
    plot_df = merged_ddf.compute()
    print("계산 완료.")

    # 5. 시각화를 위해 데이터 정렬 및 인덱스 설정
    plot_df = plot_df.set_index('name').sort_values('count', ascending=False)
    
    # 6. 시각화 (이후 로직은 이전과 거의 동일)
    plot_height = max(8, len(plot_df) * 0.4)
    fig, ax = plt.subplots(figsize=(12, plot_height))
    
    sns.barplot(x=plot_df['count'], y=plot_df.index, orient='h', palette='viridis', ax=ax)
    
    if x_log_scale:
        ax.set_xscale('log')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Annotations', fontsize=12)
    ax.set_ylabel('Class Name', fontsize=12)

    if show_counts:
        for index, value in enumerate(plot_df['count']):
            ax.text(value, index, f' {value}', va='center', fontsize=10)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
    return fig, ax

def plot_top_bottom_classes(
    annotations_ddf: dd.DataFrame,
    categories_df: pd.DataFrame,
    n_items: int = 20,
    save_path: Optional[Path] = None
):
    """상위 N개와 하위 N개 클래스 분포를 함께 그립니다."""
    # 1. Dask로 계산 계획 수립
    class_counts = annotations_ddf['category_id'].value_counts()
    top_n = class_counts.nlargest(n=n_items).to_frame(name='count')
    bottom_n = class_counts.nsmallest(n=n_items).to_frame(name='count')
    
    top_merged = top_n.merge(categories_df, left_index=True, right_index=True)
    bottom_merged = bottom_n.merge(categories_df, left_index=True, right_index=True)
    
    # 2. 실제 계산 실행
    top_df, bottom_df = dask.compute(top_merged, bottom_merged)
    print("계산 완료.")

    # 3. 시각화
    top_df = top_df.set_index('name').sort_values('count', ascending=False)
    bottom_df = bottom_df.set_index('name').sort_values('count', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, n_items * 0.4)))
    
    sns.barplot(x=top_df['count'], y=top_df.index, palette='viridis', ax=axes[0])
    axes[0].set_title(f'Top {n_items} Most Frequent Classes')
    axes[0].set_xlabel('Number of Annotations')
    
    sns.barplot(x=bottom_df['count'], y=bottom_df.index, palette='viridis_r', ax=axes[1])
    axes[1].set_title(f'Bottom {n_items} Least Frequent Classes')
    axes[1].set_xlabel('Number of Annotations')
    axes[1].yaxis.tick_right()
    
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig, axes


def plot_cumulative_class_distribution(
    ddf: dd.DataFrame,
    save_path: Optional[Path] = None
):
    """클래스 누적 분포 곡선을 그립니다."""
    print("누적 분포 계산 중...")
    class_counts = ddf['category_id'].value_counts().compute()
    print("계산 완료.")

    cumulative_dist = class_counts.cumsum() / class_counts.sum() * 100
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(np.arange(1, len(cumulative_dist) + 1), cumulative_dist.values)
    
    # 80% 지점 등 중요 지표 표시
    classes_for_80_percent = (cumulative_dist < 80).sum() + 1
    ax.axhline(80, color='r', linestyle='--', label='80% of Data')
    ax.axvline(classes_for_80_percent, color='r', linestyle='--')
    
    ax.set_title('Cumulative Distribution of Annotations by Class')
    ax.set_xlabel('Number of Classes (sorted by frequency)')
    ax.set_ylabel('Cumulative Percentage of Annotations (%)')
    ax.grid(True)
    ax.legend()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def calculate_dask_boxplot_stats(
    ddf: dd.DataFrame,
    categories_df: pd.DataFrame,
    group_col: str = 'category_id',
    value_col: str = 'area',
    top_n: Optional[int] = None
) -> pd.DataFrame:
    """
    Dask 데이터프레임에서 박스 플롯을 그리기 위한 통계치를 계산합니다.
    
    Args:
        ...
        top_n (int, optional): 상위 n개 카테고리만 선택합니다. 
                               None으로 지정하면 전체 카테고리를 대상으로 계산합니다.
    """
    print(f"'{value_col}'에 대한 박스 플롯 통계치 계산을 시작합니다...")
    
    # top_n이 정수 값일 때만 필터링이 실행됩니다.
    # None이면 이 블록을 건너뛰고 전체 ddf를 사용합니다.
    if top_n:
        top_categories = ddf[group_col].value_counts().nlargest(top_n).index.compute()
        filtered_ddf = ddf[ddf[group_col].isin(top_categories)]
    else:
        print("전체 카테고리를 대상으로 계산합니다.")
        filtered_ddf = ddf

    # 2. Dask로 Quantile 계산
    stats_series_ddf = filtered_ddf.groupby(group_col)[value_col].apply(
        lambda x: x.quantile([0.25, 0.50, 0.75]),
        meta=pd.Series(dtype='float64', name=value_col)
    )
    stats_series_pdf = stats_series_ddf.compute()
    stats_pdf = stats_series_pdf.unstack().rename(columns={0.25: 'q1', 0.50: 'med', 0.75: 'q3'})

    # 3. Whisker (수염) 계산
    stats_pdf = pd.merge(stats_pdf, categories_df, left_index=True, right_on='id')
    iqr = stats_pdf['q3'] - stats_pdf['q1']
    stats_pdf['whislo'] = (stats_pdf['q1'] - 1.5 * iqr).clip(lower=0)
    stats_pdf['whishi'] = stats_pdf['q3'] + 1.5 * iqr
    
    print("통계치 계산 완료.")
    return stats_pdf

def plot_boxplot_from_stats(
    stats_df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    xlog: bool = True,
    save_path: Optional[Path] = None
) -> tuple:
    """
    미리 계산된 통계치 데이터프레임으로 가로 방향의 박스 플롯을 그립니다.
    """
    print("박스 플롯 시각화 생성 중...")
    
    plot_stats = []
    # 중앙값(med) 기준으로 정렬하여 보기 좋게 만듬
    sorted_df = stats_df.sort_values('med', ascending=True) # 가로 방향이므로 오름차순으로 변경
    
    for _, row in sorted_df.iterrows():
        plot_stats.append({
            'label': row['name'],
            'med': row['med'], 'q1': row['q1'], 'q3': row['q3'],
            'whislo': row['whislo'], 'whishi': row['whishi'],
            'fliers': []
        })

    # 1. 카테고리 수에 따라 그래프 높이를 동적으로 조절
    num_categories = len(stats_df)
    plot_height = max(10, num_categories * 0.3) # 카테고리당 0.3인치 할당 (최소 10인치)
    
    fig, ax = plt.subplots(figsize=(16, plot_height))
    
    # 2. vert=False 옵션으로 가로 박스 플롯 생성
    ax.bxp(plot_stats, showfliers=False, vert=False, patch_artist=True)
    
    # 3. x축을 로그 스케일로 변경
    if xlog:
        ax.set_xscale('log')
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12) # x 라벨
    ax.set_ylabel(ylabel, fontsize=12) # y 라벨
    ax.grid(True, axis='x', ls="--", linewidth=0.5) # x축에만 그리드 표시

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        
    return fig, ax

def plot_dask_wh_scatter(
    ddf: dd.DataFrame,
    title: str,
    sample_frac: float = 0.1, # 10%만 샘플링하여 사용 (데이터가 크면 더 줄이세요)
    alpha: float = 0.3,
    save_path: Optional[Path] = None
) -> tuple:
    """
    Dask 데이터프레임의 너비(w)와 높이(h)를 2D 산점도(scatter plot)로 그립니다.
    데이터가 클 경우 일부만 샘플링하여 시각화합니다.
    """
    print(f"전체 데이터의 {sample_frac*100:.0f}%를 샘플링하여 너비/높이 분포를 계산합니다...")
    
    # 1. 데이터 샘플링 및 계산
    if sample_frac < 1.0:
        sampled_ddf = ddf[['w', 'h']].sample(frac=sample_frac)
    else:
        sampled_ddf = ddf[['w', 'h']]
    
    plot_df = sampled_ddf.compute()
    print("계산 완료.")

    # 2. 시각화
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=plot_df, x='w', y='h', alpha=alpha, ax=ax, edgecolor='none')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Width', fontsize=12)
    ax.set_ylabel('Height', fontsize=12)
    ax.grid(True, ls="--", linewidth=0.5)
    
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax