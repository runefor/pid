# %%
# --- 1. 라이브러리 및 기본 경로 설정 ---
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import os
import ast
import dask
import cv2
import json
from IPython.display import display
import functools
import operator

# %%
# --- 2. 데이터 로딩 ---
# 전처리된 Parquet 파일이 저장된 경로를 지정합니다.
base_dir = Path(os.getcwd()).resolve().parent
preprocessed_path = base_dir / "preprocessed_data/TL_prepro"
data_path = base_dir / "assets"

print("전처리된 Parquet 파일 로딩 중...")
images_ddf = dd.read_parquet(preprocessed_path / "*/*_images.parquet")
annotations_ddf = dd.read_parquet(preprocessed_path / "*/*_annotations.parquet")

# 카테고리 정보 로드 (이전과 동일)
with (data_path / "categories.json").open("r", encoding="utf-8") as f:
    categories = json.load(f)
categories_df = pd.DataFrame(categories)
categories_df[['category_group', 'category_name']] = categories_df['name'].str.split('@', n=1, expand=True)
print("데이터 로딩 완료.")


# %%
# display(images_ddf.compute().head())
display(annotations_ddf.compute().head())

# %%
# --- 3. 특성 엔지니어링 (w, h, area, aspect_ratio 등 계산) ---

def calculate_features(bbox_list):
    # bbox 리스트가 유효한지 확인
    if len(bbox_list) == 4:
        x, y, w, h = bbox_list
        x2 = x + w
        y2 = y + h
        aspect_ratio = w / (h + 1e-6) # 0으로 나누기 방지
        center_x = (x + w) / 2
        center_y = (y + h) / 2
        return pd.Series([x, y, w, h, x2, y2, aspect_ratio, center_x, center_y], index=['x', 'y', 'w', 'h', 'x2', 'y2','aspect_ratio', 'center_x', 'center_y'])
    else:
        # 유효하지 않은 데이터는 NaN으로 채워진 Series 반환
        return pd.Series([None] * 9, index=['x', 'y', 'w', 'h', 'x2', 'y2', 'aspect_ratio', 'center_x', 'center_y'])

# .apply 결과가 여러 컬럼으로 확장될 것을 meta로 알려줌
new_cols_ddf = annotations_ddf['bbox'].apply(
    calculate_features,
    # meta 정보에 새로 생길 컬럼들의 이름과 데이터 타입을 명시
    meta={'x': 'f8', 'y': 'f8', 'w': 'f8', 'h': 'f8', 'x2': 'f8', 'y2': 'f8', 'aspect_ratio': 'f8', 'center_x': 'f8', 'center_y': 'f8'}
)

# 기존 데이터프레임과 새로운 컬럼들을 결합
annotations_ddf = dd.concat([annotations_ddf, new_cols_ddf], axis=1)

# 결과 확인
# display(annotations_ddf.head())


# %% [markdown]
# ## 분석 1: 객체의 크기 및 형태 분포
# 전체 데이터셋에 있는 객체들의 너비(w), 높이(h), 면적(area), 종횡비(aspect_ratio) 분포를 확인합니다.

# %%
def plot_shape_distributions(ddf):
    features_to_plot = ['w', 'h', 'area', 'aspect_ratio']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    print("크기/형태 히스토그램 계산 및 시각화 생성 중...")
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        valid_data = ddf[feature].dropna()
        valid_data = valid_data[~valid_data.isin([np.inf, -np.inf])]
        
        min_val, max_val = dask.compute(valid_data.min(), valid_data.max())
        
        if feature == 'area' and min_val > 0:
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 100)
            ax.set_xscale('log')
        else:
            if feature == 'aspect_ratio':
                min_val, max_val = max(0, min_val), min(5, max_val)
            bins = np.linspace(min_val, max_val, 100)
        
        counts, bin_edges = da.histogram(valid_data.values, bins=bins)
        computed_counts, computed_bin_edges = dask.compute(counts, bin_edges)
        
        ax.bar(computed_bin_edges[:-1], computed_counts, width=np.diff(computed_bin_edges), align='edge')
        ax.set_title(f'Distribution of {feature.capitalize()}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')

    fig.tight_layout()
    plt.show()

plot_shape_distributions(annotations_ddf)

# %% [markdown]
# ## 분석 2: 클래스별 객체 크기 분포
# 가장 빈도가 높은 15개 클래스에 대해 객체 면적(area)의 분포를 박스 플롯으로 비교합니다.


# %%
# --- 1. Top 15 카테고리 ID 목록을 다시 계산 (상태 초기화) ---
top_15_categories = annotations_ddf['category_id'].value_counts().nlargest(15).compute()
top_15_ids = top_15_categories.index.tolist()
print("Top 15 카테고리 ID:", top_15_ids)

# --- 2. 데이터 타입을 '명시적으로' 일치시키기 (⭐핵심 안정화 단계⭐) ---
# top_15_ids가 정수(int) 리스트임을 확인하고,
# annotations_ddf['category_id']도 정수 타입으로 확실하게 변환합니다.
# 이렇게 하면 보이지 않는 타입 불일치 문제를 방지할 수 있습니다.
try:
    # 혹시 모를 결측치(NaN)가 문제를 일으킬 수 있으므로 먼저 제거
    annotations_ddf = annotations_ddf.dropna(subset=['category_id'])
    annotations_ddf['category_id'] = annotations_ddf['category_id'].astype(int)
    
    # id 리스트도 만약을 위해 정수로 변환
    top_15_ids = [int(i) for i in top_15_ids]

    print(f"\n데이터 타입 일치 완료: {annotations_ddf['category_id'].dtype}")
except Exception as e:
    print(f"데이터 타입을 변환하는 중 오류 발생: {e}")

# --- 3. 최적화된 .isin() 메소드 사용 ---
print("\nDask의 .isin()으로 필터링 실행 중...")
filtered_ddf = annotations_ddf[annotations_ddf['category_id'].isin(top_15_ids)]

# --- 4. 결과 확인 ---
# compute()를 호출하기 전에 len()으로 먼저 예상 길이를 확인하는 것이 안전합니다.
result_len = len(filtered_ddf)
print(f"필터링 후 데이터 개수: {result_len}")

if result_len > 0:
    print("성공: .isin() 필터링이 정상적으로 동작했습니다.")
    # 이제 이 filtered_ddf를 다음 단계(groupby)에 사용하시면 됩니다.
    # 예: display(filtered_ddf.head())
else:
    print("오류: .isin() 필터링 후에도 결과가 0입니다. 데이터에 해당 ID가 없는지 재확인이 필요합니다.")




# %%
def plot_area_by_category(anno_ddf, cat_df, n_top=15):
    # --- 1. Top 15 카테고리 빈도수 계산 ---
    print(f"Top {n_top} 카테고리 계산 중...")
    top_categories = anno_ddf['category_id'].value_counts().nlargest(n_top).compute()
    top_ids = top_categories.index.tolist()
    print("계산 완료.")

    # --- 2. Dask로 '모든' 카테고리에 대한 통계치 계산 ---
    print("\nDask로 모든 카테고리 통계치 계산 중...")
    stats_series_ddf = anno_ddf.groupby('category_id')['area'].apply(
        lambda x: x.quantile([0.25, 0.50, 0.75]),
        meta=pd.Series(dtype='float64', name='area')
    )
    stats_series_pdf = stats_series_ddf.compute()
    print("계산 완료.")

    # --- 3. Pandas로 후처리
    # 3-1. unstack 및 컬럼명 변경
    stats_pdf = stats_series_pdf.unstack().rename(columns={0.25: 'q1', 0.50: 'med', 0.75: 'q3'})

    # 3-2. Top 15 카테고리의 통계만 먼저 선택!
    # 이 시점의 stats_pdf 인덱스는 category_id이므로, .isin(top_ids)이 정상 동작합니다.
    stats_pdf_top15 = stats_pdf[stats_pdf.index.isin(top_ids)]

    # 3-3. 필터링된 작은 DataFrame에 카테고리 이름 병합 및 whisker 계산
    stats_pdf_final = pd.merge(stats_pdf_top15, cat_df, left_index=True, right_on='id')
    iqr = stats_pdf_final['q3'] - stats_pdf_final['q1']
    stats_pdf_final['whislo'] = (stats_pdf_final['q1'] - 1.5 * iqr).clip(lower=0)
    stats_pdf_final['whishi'] = stats_pdf_final['q3'] + 1.5 * iqr

    # --- 4. Matplotlib으로 시각화 ---
    plot_stats = []
    for index, row in stats_pdf_final.iterrows():
        plot_stats.append({
            'label': row['name'], 'med': row['med'], 'q1': row['q1'], 'q3': row['q3'],
            'whislo': row['whislo'], 'whishi': row['whishi'], 'fliers': []
        })

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bxp(plot_stats, showfliers=False)
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Area Distribution by Top {n_top} Categories (Full Data)')
    plt.xlabel('Category Name')
    plt.ylabel('Area (log scale)')
    plt.tight_layout()
    plt.show()
    
    return stats_pdf_final

stats_pdf = plot_area_by_category(annotations_ddf, categories_df)

# %% [markdown]
# ## 분석 3: 이상치(Outlier) 확인
# 특정 클래스에서 통계적으로 크기가 매우 큰 이상치를 직접 확인하고 이미지에 표시합니다.

# %%
def investigate_outlier_image(anno_ddf, img_ddf, stats_df, category_name, img_root_path):
    # category_id 찾기
    cat_info = stats_df[stats_df['name'] == category_name]
    if cat_info.empty:
        print(f"'{category_name}'을 찾을 수 없습니다.")
        return
    
    # 인덱스 대신 'id' 컬럼에서 category_id를 가져옵니다.
    category_id = cat_info['id'].iloc[0]
    
    # 이상치 경계 계산
    upper_bound = cat_info['whishi'].iloc[0]
    print(f"'{category_name}' (ID: {category_id}) 클래스의 이상치 상한선: {upper_bound:.2f}")

    # 이상치 필터링
    outliers_ddf = anno_ddf[(anno_ddf['category_id'] == category_id) & (anno_ddf['area'] > upper_bound)]
    
    # 필터링 결과를 올바르게 확인하고 .compute()를 호출합니다.
    # 먼저 계산 계획을 실행하여 실제 Pandas DataFrame으로 결과를 가져옵니다.
    outlier_sample_pd = outliers_ddf.head(1)

    if outlier_sample_pd.empty:
        print("상한선을 벗어나는 이상치를 찾을 수 없습니다.")
        return

    # 이제부터 outlier_sample_pd (Pandas DataFrame)를 사용합니다.
    # 이미지 파일 이름 찾기
    image_id = outlier_sample_pd['image_id'].iloc[0]
    
    # 이미지 정보 찾기 (id가 문자열일 수 있으므로 .astype(str) 추가)
    image_info = img_ddf[img_ddf['id'].astype(str) == image_id][['file_name', 'attributes.vendor', 'attributes.shipType']].compute()
    
    if image_info.empty:
        print(f"Image ID '{image_id}'에 해당하는 파일을 찾을 수 없습니다.")
        return
        
    image_name_str = image_info['file_name'].iloc[0]
    vendor = image_info['attributes.vendor'].iloc[0]
    shipType = image_info['attributes.shipType'].iloc[0]
    
    # 이미지 경로 조합
    # image_id에서 파일명을 만들었으므로 image_name_str을 직접 사용
    image_path = img_root_path / vendor / shipType / f"{image_id}.png" # 실제 경로 구조에 맞게 수정
    
    # 이미지에 bbox 그리기
    try:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox = outlier_sample_pd['bbox'].iloc[0]
        x, y, w, h = [int(c) for c in bbox]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 50)

        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.title(f"Outlier Check: {category_name} in {image_id}.png")
        plt.show()
    except Exception as e:
        print(f"이미지를 표시하는 중 오류 발생: {e}\n경로: {image_path}")

# 함수 호출 (stats_pdf 대신 stats_pdf_final 사용을 권장)
image_root_path = data_path / 'TS' 
investigate_outlier_image(annotations_ddf, images_ddf, stats_pdf, 'Equipments@Spray Nozzle', image_root_path)
# %%
images_ddf['id'] = images_ddf['id'].astype('str')
annotations_ddf['image_id'] = annotations_ddf['image_id'].astype('str')
print("타입 통일 완료.")
# --- 1. 작은 DataFrame(images_ddf)을 Pandas로 변환 ---
print("작은 images_ddf를 메모리로 로드 중...")
# 병합 및 계산에 필요한 컬럼만 선택합니다.
images_pdf = images_ddf[['id', 'width', 'height']].rename(columns={'id': 'image_id'}).compute()
print("로드 완료.")

# --- 2. map_partitions를 이용한 병합 및 계산 ---
print("map_partitions으로 병합 및 정규화 계산 계획 수립 중...")
# 병합에 필요한 annotations 컬럼만 선택
annotations_to_merge = annotations_ddf[['image_id', 'center_x', 'center_y', 'category_id']]

def merge_and_normalize(partition_df, images_lookup_df):
    """한 파티션에 대해 병합과 정규화 계산을 모두 수행하는 함수"""
    merged_partition = pd.merge(partition_df, images_lookup_df, on='image_id')
    merged_partition['norm_center_x'] = merged_partition['center_x'] / merged_partition['width']
    merged_partition['norm_center_y'] = merged_partition['center_y'] / merged_partition['height']
    return merged_partition

# Dask에게 결과물의 구조(meta)를 알려주기
meta_df = pd.DataFrame({
    'image_id': pd.Series(dtype='str'),
    'center_x': pd.Series(dtype='float32'),
    'center_y': pd.Series(dtype='float32'),
    'category_id': pd.Series(dtype='int64'), # category_id 타입에 맞게 조정
    'width': pd.Series(dtype='int16'),
    'height': pd.Series(dtype='int16'),
    'norm_center_x': pd.Series(dtype='float64'),
    'norm_center_y': pd.Series(dtype='float64'),
})

dask_merged = annotations_to_merge.map_partitions(
    merge_and_normalize,
    images_lookup_df=images_pdf,
    meta=meta_df
)

# --- 3. 2D 히스토그램 계산 및 시각화 (이전과 동일) ---
print("시각화용 데이터 계산 중 (2D 히스토그램)...")
gridsize = 100
bins = [gridsize, gridsize]
range_ = [[0, 1], [0, 1]]

H, xedges, yedges = da.histogram2d(
    dask_merged['norm_center_x'].values,
    dask_merged['norm_center_y'].values,
    bins=bins,
    range=range_
)

computed_H, computed_xedges, computed_yedges = dask.compute(H, xedges, yedges)
print("계산 완료.")

# --- 3. 계산된 결과를 Matplotlib으로 직접 시각화 ---
print("Matplotlib으로 시각화 생성 중...")
# jointplot과 유사한 레이아웃을 직접 만듭니다.
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(4, 4)
ax_main = fig.add_subplot(gs[1:4, 0:3])
ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

# 메인 2D 히스토그램 (hexbin 대신 pcolormesh 사용)
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

# 전체 제목 및 레이아웃 조정
fig.suptitle('Normalized Spatial Distribution of Bounding Box Centers', y=0.95)
ax_main.invert_yaxis() # y축 뒤집기
gs.tight_layout(fig)
gs.update(wspace=0.05, hspace=0.05)

plt.show()
# %%
