# main_preprocess_pipeline.py
import dataclasses
import os
from pathlib import Path

import pandas as pd

from pid_preprocess.data_loader import DataLoader, setting_categories_data, LoadedDaskData
from pid_preprocess.feature_engineering import add_bbox_features
from pid_preprocess import eda_runner # eda_runner에서 visualize_combined_class_distribution을 가져옴
from utils.file_utils import load_json_data, save_json_data


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


def classes_count(annotations_df: pd.DataFrame) -> pd.Series:
    class_counts: pd.DataFrame = annotations_df['category_id'].value_counts()
    
    # print(class_counts)
    
    tasks = {
        'total_classes': class_counts.shape[0],
        'min_count': class_counts.min(),
        'max_count': class_counts.max(),
        'mean_count': class_counts.mean(),
        'median_count': class_counts.median(),
        'categories_values_counts_not_over_one': class_counts[class_counts == 1].shape[0],
        'categories_values_counts_over_one': class_counts[class_counts > 1].shape[0],
    }
    
    results = pd.Series(tasks)
    results['max_min_ratio'] = results['max_count'] / results['min_count'] if results['min_count'] > 0 else float('inf')
    
    # 5. 누적 분포 계산 (이 부분은 계산된 Pandas Series로 수행하는 것이 더 효율적)
    cumulative_percentage = class_counts.cumsum() / class_counts.sum()
    results['classes_for_50_percent'] = (cumulative_percentage < 0.50).sum() + 1
    results['classes_for_80_percent'] = (cumulative_percentage < 0.80).sum() + 1
    results['classes_for_95_percent'] = (cumulative_percentage < 0.95).sum() + 1
    
    print("계산 완료.")

    print(results)
    
    return results


def obj_list(annotations_df: pd.DataFrame):
    
    # min count가 1인 클래스가 '1745', '1753' 이렇게 두개 있어서 그걸 제외한 나머지 값들이 obj_list임.
    # 1745는 275개 존재한다고 나옴, 다만 홈페이지에는 299라는데, 라벨링 된 이미지를 보니까 약간 복잡해 보여서 뺀 것 같음. 일단 이상하게 보이니까 제외시키자!
    obj_list = ["1102", "1201", "1202", "1206", "1209", "1210", "1211", "1212", "1301", "1401", "1501", "1502", "1504", "1505",
            "1506", "1507", "1508", "1509", "1511", "1512", "1513", "1514", "1517", "1518", "1519", "1523", "1524", "1525", "1526", "1528",
            "1530", "1533", "1535", "1603", "1701", "1704", "1706", "1707", "1709", "1710", "1711", "1713", "1715", "1716", "1717", "1719",
            "1722", "1723", "1725", "1726", "1733", "1734", "1735", "1736", "1742", "1743", "1744", "1746", "1747", "1749", "1751", "1752",
            "1754", "1755", "1757", "1758", "1759", "1801", "1810", "1812", "1813", "1903", "1907", "1908", "1909", "1913", "1920", "1921",
            "1922", "1927", "1933", "1934", "1935", "1937", "1938", "1942", "1946", "1947", "1951", "1952", "1954", "1955", "1956", "1958",
            "1962", "1963", "1965", "1966", "1967", "1968", "1969", "2001", "2102", "2103"]
    # print(annotations_df['category_id'].value_counts())
    print(annotations_df["attributes"].apply(lambda x: x.get("vendor")).value_counts())
    # print(annotations_df["attributes"][annotations_df["attributes"]["vendor"] != "V01"])
    
    obj_set = set(obj_list)
    print(f"obj 리스트의 총 개수: {len(obj_set)}")
    ann_obj_set = set(annotations_df['category_id'].unique().astype(str).tolist())
    only_in_obj = obj_set - ann_obj_set
    print(f"obj에는 있는데 어노테이션에는 없는 값: {only_in_obj}")
    print(f"Number of only in obj: {len(only_in_obj)}")
    only_in_df = ann_obj_set - obj_set
    print(f"어노테이션에는 있는데 obj에는 없는 값: {only_in_df}")
    print(f"Number of non-overlapping classes: {len(only_in_df)}")
    print((annotations_df["category_id"] == 1745).sum())
    

def save_except_one_categories(coco_data: dict, save_path: Path, categories_id: int=1745, file_name: str="merged_v01_prepro.json") -> None:
    annotations_df = pd.DataFrame(coco_data['annotations'])
    filtered_annotations_df = annotations_df[annotations_df['category_id'] != categories_id]
    coco_data["annotations"] = filtered_annotations_df.to_dict(orient="records")

    # Filter categories to keep only those present in the filtered annotations
    active_category_ids = set(filtered_annotations_df['category_id'].unique())
    original_categories = coco_data['categories']
    filtered_categories = [cat for cat in original_categories if cat['id'] in active_category_ids]
    coco_data['categories'] = filtered_categories

    save_file = save_path / file_name
    save_json_data(coco_data, save_file)
    print(f"Saved except category_id {categories_id} to {save_file}")
    print(f"Original category count: {len(original_categories)}, Final category count: {len(filtered_categories)}")


def save_categories_values_counts_over_one(coco_data: dict, save_path: Path, file_name: str="merged_v01_prepro.json") -> None:
    annotations_df = pd.DataFrame(coco_data['annotations'])
    counts = annotations_df['category_id'].value_counts()
    # filtered_annotations  = annotations_df.groupby('category_id').filter(lambda x: len(x) > 1)
    filtered_annotations_df = annotations_df[annotations_df['category_id'].isin(counts[counts > 1].index)]
    coco_data["annotations"] = filtered_annotations_df.to_dict(orient="records")

    # Filter categories to keep only those present in the filtered annotations
    active_category_ids = set(filtered_annotations_df['category_id'].unique())
    original_categories = coco_data['categories']
    filtered_categories = [cat for cat in original_categories if cat['id'] in active_category_ids]
    coco_data['categories'] = filtered_categories

    save_file = save_path / file_name
    save_json_data(coco_data, save_file)
    print(f"Saved categories with counts over one to {save_file}")
    print(f"Original category count: {len(original_categories)}, Final category count: {len(filtered_categories)}")


def data_preprocess() -> None:
    base_dir = Path(os.getcwd()).resolve()
    data_path = base_dir / "assets"
    
    save_path = base_dir / f"reports_{ver}"
    
    # coco_data = load_json_data(data_path / "merged_dataset.json")
    # coco_data = load_json_data(data_path / "merged_v01_dataset.json")
    coco_data = load_json_data(data_path / "merged_v01_prepro.json")
    
    categories_df = pd.DataFrame(coco_data['categories'])
    annotations_df = pd.DataFrame(coco_data['annotations'])
    images_df = pd.DataFrame(coco_data['images'])
    
    print(annotations_df.info())
    
    print("---------------------------")
    
    print(images_df.info())
    
    # filtered_df = annotations_df[annotations_df["attributes"].apply(lambda x: x.get("vendor") == "V01")]
    
    # classes_count(annotations_df)
    # obj_list(annotations_df)
    save_except_one_categories(coco_data, data_path, categories_id=1745, file_name="merged_v01_prepro.json")
    # save_categories_values_counts_over_one(coco_data, data_path, file_name="merged_v01_prepro.json")
    
    
# --- Temporary EDA functions (Pandas version) ---
def write_eda_image_report(save_path: Path, title: str, image_name: str, good_desc: str, bad_desc: str, analysis_desc: str):
    """Helper function to generate a markdown report for an EDA image."""
    md_path = save_path.with_suffix('.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {title}\n\n")
        f.write(f"![{title}]({image_name})\n\n")
        f.write("---\n\n")
        f.write("## 결과 해석 (Interpreting the Results)\n\n")
        f.write("### 좋은 결과 (Good Result)\n")
        f.write(f"- {good_desc}\n\n")
        f.write("### 나쁜 결과 (Bad Result)\n")
        f.write(f"- {bad_desc}\n\n")
        f.write("### 현재 데이터 분석\n")
        f.write(f"- {analysis_desc}\n")
    print(f"Saved analysis report to {md_path}")

def temp_add_bbox_features(annotations_df: pd.DataFrame) -> pd.DataFrame:
    if 'bbox' not in annotations_df.columns or annotations_df['bbox'].isnull().all():
        print("Bbox column not found or is empty, skipping feature creation.")
        return annotations_df
    
    df = annotations_df.copy()
    df = df[df['bbox'].apply(lambda x: isinstance(x, list) and len(x) == 4)]
    
    df['bbox_x'] = df['bbox'].apply(lambda b: b[0])
    df['bbox_y'] = df['bbox'].apply(lambda b: b[1])
    df['bbox_width'] = df['bbox'].apply(lambda b: b[2])
    df['bbox_height'] = df['bbox'].apply(lambda b: b[3])
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    df['aspect_ratio'] = df['bbox_width'] / (df['bbox_height'] + 1e-6)
    return df

def temp_run_bbox_analysis(annotations_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    print(f"[{prefix}] Running bbox analysis...")
    if annotations_df.empty or 'bbox_area' not in annotations_df.columns:
        print(f"[{prefix}] Annotations are empty or bbox features are missing, skipping bbox analysis.")
        return

    # Plot bbox area distribution
    plt.figure(figsize=(10, 6))
    plt.hist(annotations_df['bbox_area'], bins=50, log=True)
    plt.xlabel("Bbox Area (pixels^2)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Bbox Area Distribution for '{prefix}'")
    plot_path = save_path / f"{prefix}_bbox_area_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved bbox area plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="바운딩 박스 넓이(Area) 분포",
        image_name=plot_path.name,
        good_desc="객체들의 크기가 비교적 일정하여 분포의 폭이 좁고, 특정 영역에 집중된 분포를 보입니다.",
        bad_desc="분포가 매우 넓게 퍼져있거나, 극단적으로 작은 객체들이 너무 많은 경우입니다. 이는 모델이 다양한 크기의 객체를 모두 잘 학습하기 어렵게 만듭니다.",
        analysis_desc="로그 스케일(log scale)의 히스토그램은 객체 크기의 전반적인 분포를 보여줍니다. 그래프가 왼쪽에 치우쳐 있다면 데이터셋에 작은 객체가 매우 많다는 의미이며, 이는 작은 객체 탐지 성능에 중요한 영향을 미칩니다."
    )

    # Plot aspect ratio distribution
    plt.figure(figsize=(10, 6))
    plt.hist(annotations_df['aspect_ratio'], bins=50, range=(0, 5))
    plt.xlabel("Aspect Ratio (width/height)")
    plt.ylabel("Frequency")
    plt.title(f"Bbox Aspect Ratio Distribution for '{prefix}'")
    plot_path = save_path / f"{prefix}_aspect_ratio_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved aspect ratio plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="바운딩 박스 종횡비(Aspect Ratio) 분포",
        image_name=plot_path.name,
        good_desc="분포가 1.0 근처(정사각형)에 집중되거나, 가로/세로 방향성을 가진 몇 개의 그룹으로 명확히 나뉩니다.",
        bad_desc="분포가 매우 넓고 평평하게 퍼져 있어 객체들의 형태가 매우 다양하고 예측하기 어렵다는 것을 의미합니다.",
        analysis_desc="객체들의 가로/세로 비율 분포를 보여줍니다. 1.0을 기준으로 좌우대칭에 가까우면 가로/세로로 긴 객체들이 균형있게 존재한다는 의미입니다. 특정 지점에 피크(peak)가 있다면 해당 형태의 객체가 많다는 뜻이며, 이는 앵커 박스 설계에 중요한 정보가 됩니다."
    )

def temp_run_class_distribution_analysis(annotations_df: pd.DataFrame, categories_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    print(f"[{prefix}] Running class distribution analysis...")
    if annotations_df.empty:
        print(f"[{prefix}] Annotations are empty, skipping class distribution analysis.")
        return

    class_counts = annotations_df['category_id'].value_counts()
    class_counts_df = class_counts.reset_index()
    class_counts_df.columns = ['category_id', 'count']
    class_counts_df = class_counts_df.merge(categories_df, left_on='category_id', right_on='id', how='left')
    class_counts_df = class_counts_df.sort_values(by='count', ascending=False)
    
    summary_path = save_path / f"{prefix}_class_distribution_summary.md"
    max_count = class_counts.max()
    min_count = class_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(class_counts_df[['name', 'count']].to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 결과 해석 (Interpreting the Results)\n\n")
        f.write("### 좋은 결과 (Good Result)\n")
        f.write("- **균등한 분포**: 모든 클래스의 객체 수가 비교적 균등하게 분포합니다. (e.g., 최다/최소 클래스 비율이 10:1 미만)\n")
        f.write("- **영향**: 클래스 불균형이 적으면 모델이 모든 클래스를 편향 없이 공평하게 학습할 가능성이 높습니다.\n\n")
        f.write("### 나쁜 결과 (Bad Result)\n")
        f.write("- **심한 불균형 (Long-tail 분포)**: 특정 소수 클래스가 대부분의 데이터를 차지하고, 나머지 대다수 클래스의 데이터는 매우 적습니다. (e.g., 최다/최소 클래스 비율이 100:1 이상)\n")
        f.write("- **영향**: 모델이 데이터가 많은 클래스에만 과적합(overfitting)되고, 데이터가 적은 소수 클래스는 거의 탐지하지 못하는 성능 저하를 유발합니다.\n\n")
        f.write("### 현재 데이터 분석\n")
        f.write(f"- 현재 데이터의 최다/최소 클래스 간 객체 수 비율은 **{imbalance_ratio:.1f}:1** 입니다.\n")
        if imbalance_ratio > 100:
            f.write("- **판단**: 데이터 불균형이 매우 심각한 'Long-tail' 분포를 보입니다. 소수 클래스에 대한 데이터 증강(Augmentation), 리샘플링(Resampling), 또는 손실 함수(Loss function) 가중치 조절 등의 전략이 반드시 필요합니다.\n")
        elif imbalance_ratio > 10:
            f.write("- **판단**: 데이터 불균형이 다소 존재합니다. 소수 클래스에 대한 성능 저하가 우려되므로 관련 전략을 고려하는 것이 좋습니다.\n")
        else:
            f.write("- **판단**: 데이터 분포가 비교적 양호합니다.\n")
    print(f"Saved class distribution summary and interpretation to {summary_path}")

    plt.figure(figsize=(20, 10))
    top_n = 50
    plt.bar(class_counts_df['name'].head(top_n), class_counts_df['count'].head(top_n))
    plt.xlabel("Category")
    plt.ylabel("Object Count")
    plt.title(f"Class Distribution for '{prefix}' (Top {top_n})")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_path = save_path / f"{prefix}_class_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved class distribution plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="클래스 분포",
        image_name=plot_path.name,
        good_desc="막대그래프의 높이가 전반적으로 균일하여 모든 클래스가 비슷한 수의 데이터를 가집니다.",
        bad_desc="몇몇 클래스의 막대가 비정상적으로 높고 나머지 대다수 클래스의 막대는 매우 낮은 'Long-tail' 형태를 보입니다.",
        analysis_desc="가장 데이터가 많은 Top 50개 클래스의 객체 수를 보여줍니다. 이 그래프를 통해 어떤 클래스가 데이터셋을 지배하고 있는지, 어떤 클래스가 희귀한지 직관적으로 파악할 수 있습니다. 데이터 불균형의 정도를 시각적으로 확인할 수 있습니다."
    )

def temp_run_image_property_analysis(images_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    print(f"[{prefix}] Running image property analysis...")
    if images_df.empty:
        print(f"[{prefix}] Images are empty, skipping image property analysis.")
        return

    resolutions = images_df[['width', 'height']].value_counts().reset_index(name='count')
    
    summary_path = save_path / f"{prefix}_resolution_summary.md"
    num_unique_resolutions = len(resolutions)
    is_varied = num_unique_resolutions > 50

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(resolutions.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write("## 결과 해석 (Interpreting the Results)\n\n")
        f.write("### 좋은 결과 (Good Result)\n")
        f.write("- **일관된 해상도**: 이미지 해상도가 대부분 일정하거나, 몇 개의 그룹으로 명확히 군집화됩니다.\n")
        f.write("- **영향**: 해상도가 일정하면 모델의 입력 크기를 정하고 모든 이미지에 일관된 리사이즈(resize) 전략을 적용하기 용이합니다.\n\n")
        f.write("### 나쁜 결과 (Bad Result)\n")
        f.write("- **지나치게 다양한 해상도**: 해상도가 특정 패턴 없이 매우 다양하게 퍼져 있습니다.\n")
        f.write("- **영향**: 일관성 없는 리사이즈 전략은 객체의 왜곡을 유발할 수 있습니다. 특히 작은 객체는 리사이즈 과정에서 정보가 소실되어 탐지가 더 어려워질 수 있습니다.\n\n")
        f.write("### 현재 데이터 분석\n")
        f.write(f"- 현재 데이터에는 **{num_unique_resolutions}개**의 고유한 이미지 해상도가 존재합니다.\n")
        if is_varied:
            f.write("- **판단**: 이미지 해상도가 매우 다양합니다. 모델 학습 시 이미지 크기를 통일하는 과정(e.g., padding, resizing)에서 원본 객체의 비율이 왜곡되지 않도록 주의 깊은 전처리 전략이 필요합니다.\n")
        else:
            f.write("- **판단**: 이미지 해상도가 비교적 일관되거나 몇 개의 그룹으로 군집되어 있습니다. 전처리 전략을 세우기 용이한 편입니다.\n")
    print(f"Saved resolution summary and interpretation to {summary_path}")

    plt.figure(figsize=(10, 8))
    plt.scatter(resolutions['width'], resolutions['height'], s=resolutions['count'], alpha=0.5)
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title(f"Image Resolution Distribution for '{prefix}' (bubble size = count)")
    plt.grid(True)
    plot_path = save_path / f"{prefix}_resolution_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved resolution plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="이미지 해상도 분포",
        image_name=plot_path.name,
        good_desc="점들이 몇 개의 명확한 클러스터(군집)를 형성합니다.",
        bad_desc="점들이 특정 패턴 없이 넓은 영역에 무작위로 흩어져 있습니다.",
        analysis_desc="데이터셋에 포함된 이미지들의 가로/세로 해상도 분포를 보여줍니다. 점의 크기는 해당 해상도를 가진 이미지의 수를 의미합니다. 이 그래프를 통해 데이터셋의 해상도 편향성을 파악하고, 모델 입력 크기 선정 및 리사이즈 정책 수립에 활용할 수 있습니다."
    )

def temp_run_object_cooccurrence_analysis(annotations_df: pd.DataFrame, categories_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(f"[{prefix}] Running object co-occurrence analysis...")
    if annotations_df.empty:
        print(f"[{prefix}] Annotations are empty, skipping co-occurrence analysis.")
        return

    image_to_cats = annotations_df.groupby('image_id')['category_id'].unique()
    cat_ids = sorted(categories_df['id'].unique().tolist())
    cat_id_to_name = pd.Series(categories_df.name.values, index=categories_df.id).to_dict()
    
    co_occurrence = pd.DataFrame(0, index=cat_ids, columns=cat_ids)

    for cats_in_image in image_to_cats:
        for i in range(len(cats_in_image)):
            for j in range(i, len(cats_in_image)):
                cat1 = cats_in_image[i]
                cat2 = cats_in_image[j]
                if cat1 in co_occurrence.index and cat2 in co_occurrence.columns:
                    co_occurrence.loc[cat1, cat2] += 1
                    if cat1 != cat2:
                        co_occurrence.loc[cat2, cat1] += 1
    
    top_n = 25
    top_n_cat_ids = annotations_df['category_id'].value_counts().nlargest(top_n).index
    
    co_occurrence_top_n = co_occurrence.loc[top_n_cat_ids, top_n_cat_ids]
    co_occurrence_top_n.rename(index=cat_id_to_name, columns=cat_id_to_name, inplace=True)

    plt.figure(figsize=(15, 15))
    sns.heatmap(co_occurrence_top_n, annot=True, fmt='d', cmap='viridis')
    plt.title(f"Object Co-occurrence Heatmap for '{prefix}' (Top {top_n} Classes)")
    plt.tight_layout()
    plot_path = save_path / f"{prefix}_co_occurrence_heatmap.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved co-occurrence heatmap to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="객체 동시 발생(Co-occurrence) 분석",
        image_name=plot_path.name,
        good_desc="(판단 기준 없음) 데이터의 고유한 특성을 나타내며, 패턴을 발견하는 것이 중요합니다.",
        bad_desc="(판단 기준 없음) 대각선 외의 모든 셀이 어두운 색이라면 클래스 간의 관계성이 거의 없다는 의미입니다.",
        analysis_desc="두 개의 다른 클래스가 얼마나 자주 같은 이미지에 나타나는지를 보여주는 히트맵입니다. Top 25개 클래스에 대해 분석하며, 대각선은 각 클래스의 총 등장 횟수를 의미합니다. 대각선 외의 밝은 셀은 함께 등장하는 빈도가 높은 클래스 쌍을 나타내며, 이를 통해 데이터의 문맥적 관계를 파악할 수 있습니다."
    )

def temp_run_spatial_distribution_analysis(annotations_df: pd.DataFrame, images_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    print(f"[{prefix}] Running spatial distribution analysis...")
    if annotations_df.empty or images_df.empty or 'bbox_x' not in annotations_df.columns:
        print(f"[{prefix}] Data is missing for spatial distribution analysis, skipping.")
        return

    merged_df = pd.merge(annotations_df, images_df, left_on='image_id', right_on='id', how='inner')
    
    if 'width' not in merged_df.columns or 'height' not in merged_df.columns:
        print(f"[{prefix}] Image dimensions are missing, cannot run spatial analysis.")
        return

    center_x_norm = (merged_df['bbox_x'] + merged_df['bbox_width'] / 2) / merged_df['width']
    center_y_norm = (merged_df['bbox_y'] + merged_df['bbox_height'] / 2) / merged_df['height']

    plt.figure(figsize=(8, 8))
    plt.hist2d(center_x_norm, center_y_norm, bins=50, cmap='inferno')
    plt.xlabel("Normalized X-coordinate")
    plt.ylabel("Normalized Y-coordinate")
    plt.title(f"Spatial Distribution of Object Centers for '{prefix}'")
    plt.colorbar(label='Object Count')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plot_path = save_path / f"{prefix}_spatial_distribution.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved spatial distribution plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="객체 위치 분포 분석",
        image_name=plot_path.name,
        good_desc="객체들이 이미지 전체에 비교적 고르게 분포되어 있습니다.",
        bad_desc="객체들이 특정 위치(e.g., 중앙, 특정 코너)에만 집중적으로 나타나는 강한 편향을 보입니다.",
        analysis_desc="이미지를 0~1 사이로 정규화했을 때, 객체들의 중심점이 주로 어디에 위치하는지 보여주는 2D 히스토그램입니다. 노란색에 가까울수록 해당 위치에 객체가 많이 밀집되어 있다는 의미입니다. 이를 통해 데이터 수집 과정에서의 편향을 감지하거나, 특정 위치를 크롭(crop)하는 등의 증강 전략을 세울 수 있습니다."
    )

def temp_run_per_class_bbox_analysis(annotations_df: pd.DataFrame, categories_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(f"[{prefix}] Running per-class bbox analysis...")
    if annotations_df.empty or 'bbox_area' not in annotations_df.columns:
        print(f"[{prefix}] Annotations or bbox features are missing, skipping per-class bbox analysis.")
        return

    top_n = 20
    top_n_cat_ids = annotations_df['category_id'].value_counts().nlargest(top_n).index
    
    df_top_n = annotations_df[annotations_df['category_id'].isin(top_n_cat_ids)]
    df_top_n = df_top_n.merge(categories_df, left_on='category_id', right_on='id')

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df_top_n, x='name', y='bbox_area', showfliers=False)
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.title(f"Bbox Area Distribution per Class (Top {top_n})")
    plt.tight_layout()
    plot_path = save_path / f"{prefix}_per_class_bbox_area.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved per-class bbox area plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="클래스별 바운딩 박스 넓이 분석",
        image_name=plot_path.name,
        good_desc="각 클래스의 박스(box) 길이가 짧아, 클래스 내 객체들의 크기가 일정합니다.",
        bad_desc="특정 클래스의 박스 길이가 매우 길어, 해당 클래스는 다양한 크기의 객체를 포함하고 있어 모델이 학습하기 어렵습니다.",
        analysis_desc="Top 20개 클래스에 대해, 각 클래스별 객체 넓이의 분포를 박스 플롯으로 보여줍니다. 이를 통해 어떤 클래스가 항상 작은지, 어떤 클래스가 크기 변화가 심한지 등을 파악하여 탐지가 어려운 클래스를 예측하고, multi-scale 학습 전략 등을 세울 수 있습니다."
    )

    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df_top_n, x='name', y='aspect_ratio', showfliers=False)
    plt.xticks(rotation=90)
    plt.title(f"Aspect Ratio Distribution per Class (Top {top_n})")
    plt.tight_layout()
    plot_path = save_path / f"{prefix}_per_class_aspect_ratio.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved per-class aspect ratio plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="클래스별 바운딩 박스 종횡비 분석",
        image_name=plot_path.name,
        good_desc="각 클래스의 박스 길이가 짧아, 클래스 내 객체들의 형태(종횡비)가 일정합니다.",
        bad_desc="특정 클래스의 박스 길이가 매우 길어, 해당 클래스는 다양한 형태의 객체를 포함하고 있어 모델이 학습하기 어렵습니다.",
        analysis_desc="Top 20개 클래스에 대해, 각 클래스별 객체 종횡비의 분포를 박스 플롯으로 보여줍니다. 이를 통해 클래스별 고유의 형태적 특징(e.g., 파이프는 가로로 길다)을 파악하고, 앵커 박스 설계에 활용할 수 있습니다."
    )

def temp_run_objects_per_image_analysis(annotations_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    print(f"[{prefix}] Running objects per image analysis...")
    if annotations_df.empty:
        print(f"[{prefix}] Annotations are empty, skipping objects per image analysis.")
        return

    objects_per_image = annotations_df.groupby('image_id').size()

    plt.figure(figsize=(12, 6))
    if not objects_per_image.empty:
        plt.hist(objects_per_image, bins=max(50, objects_per_image.max()), log=True)
    plt.xlabel("Number of Objects per Image")
    plt.ylabel("Number of Images (log scale)")
    plt.title(f"Distribution of Objects per Image for '{prefix}'")
    plot_path = save_path / f"{prefix}_objects_per_image.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved objects per image plot to {plot_path}")
    write_eda_image_report(
        save_path=plot_path,
        title="이미지당 객체 수 분포 분석",
        image_name=plot_path.name,
        good_desc="분포가 특정 구간에 집중되어 있어, 대부분의 이미지가 비슷한 수의 객체를 포함합니다.",
        bad_desc="분포가 오른쪽으로 긴 꼬리를 가져, 일부 이미지에 객체가 매우 많이 몰려있는 '복잡한' 이미지가 존재합니다.",
        analysis_desc="이미지 한 장에 포함된 객체 수의 분포를 보여줍니다. 이를 통해 데이터셋이 전반적으로 희소(sparse)한지, 밀집(dense)한지 파악할 수 있습니다. 객체가 매우 많은 이미지는 탐지 모델에게 어려운 과제가 될 수 있습니다."
    )

def temp_run_minority_majority_class_analysis(annotations_df: pd.DataFrame, categories_df: pd.DataFrame, save_path: Path, prefix: str):
    import matplotlib.pyplot as plt
    import seaborn as sns
    print(f"[{prefix}] Running minority/majority class analysis...")
    if annotations_df.empty:
        print(f"[{prefix}] Annotations are empty, skipping analysis.")
        return

    class_counts = annotations_df['category_id'].value_counts()
    
    class_counts_df = class_counts.reset_index()
    class_counts_df.columns = ['category_id', 'count']
    class_counts_df = class_counts_df.merge(categories_df, left_on='category_id', right_on='id', how='left')

    majority_classes = class_counts_df.nlargest(10, 'count')
    minority_classes = class_counts_df.nsmallest(10, 'count')
    
    combined_df = pd.concat([majority_classes, minority_classes.sort_values('count', ascending=True)], ignore_index=True)

    plt.figure(figsize=(12, 10))
    sns.barplot(data=combined_df, x='count', y='name', hue='name', dodge=False, legend=False)
    plt.xlabel("Object Count")
    plt.ylabel("Category")
    plt.xscale('log')
    plt.title(f"Top 10 Majority vs. Bottom 10 Minority Classes for '{prefix}'")
    plt.tight_layout()
    
    plot_path = save_path / f"{prefix}_minority_majority_classes.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved minority/majority class plot to {plot_path}")

    write_eda_image_report(
        save_path=plot_path,
        title="다수/소수 클래스 분포 분석",
        image_name=plot_path.name,
        good_desc="다수 클래스와 소수 클래스 간의 객체 수 차이가 크지 않습니다 (e.g., 10:1 미만).",
        bad_desc="다수 클래스와 소수 클래스 간의 객체 수 차이가 극심하여, 그래프의 양쪽 막대 길이 차이가 매우 큽니다.",
        analysis_desc="데이터셋에서 가장 수가 많은 10개의 클래스(다수 클래스)와 가장 수가 적은 10개의 클래스(소수 클래스)를 비교하여 보여줍니다. 이 그래프는 클래스 불균형의 극단적인 정도를 명확하게 시각화하여, 모델이 어떤 클래스를 학습하기 쉽고 어떤 클래스를 어려워할지 직접적으로 보여줍니다. 소수 클래스에 대한 대책 마련이 시급한지 판단하는 데 도움이 됩니다."
    )


def data_ploting() -> None:
    print("data plotting start")

    # Path setup
    base_dir = Path(os.getcwd()).resolve()
    data_path = base_dir / "assets"
    save_path = base_dir / f"reports_{ver}_filnal"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the single merged file
    print(f"Loading merged data from {data_path / 'merged_v01_prepro.json'}")
    coco_data = load_json_data(data_path / "merged_v01_prepro.json")
    
    # 2. Prepare pandas dataframes
    print("Preparing data for analysis...")
    annotations_df = pd.DataFrame(coco_data['annotations'])
    images_df = pd.DataFrame(coco_data['images'])
    categories_df = pd.DataFrame(coco_data['categories'])
    
    # Add bbox features necessary for some analyses
    annotations_df = temp_add_bbox_features(annotations_df)

    # 3. Run all analyses on the entire dataset
    print("\nRunning base EDA analyses...")
    temp_run_bbox_analysis(annotations_df, save_path, prefix="merged")
    temp_run_class_distribution_analysis(annotations_df, categories_df, save_path, prefix="merged")
    temp_run_image_property_analysis(images_df, save_path, prefix="merged")
    
    print("\nRunning additional EDA analyses...")
    temp_run_object_cooccurrence_analysis(annotations_df, categories_df, save_path, prefix="merged")
    temp_run_spatial_distribution_analysis(annotations_df, images_df, save_path, prefix="merged")
    temp_run_per_class_bbox_analysis(annotations_df, categories_df, save_path, prefix="merged")
    temp_run_objects_per_image_analysis(annotations_df, save_path, prefix="merged")
    temp_run_minority_majority_class_analysis(annotations_df, categories_df, save_path, prefix="merged")
    
    # 4. Inform the user about the analyses that cannot be performed.
    print("\nNOTE: Train/Test comparison analyses require separate train and test datasets and are not performed.")
    
    print("\nData plotting finished.")
    
    

if __name__ == "__main__":
    # data_pipeline()
    
    data_preprocess()
    
    
    # data_ploting()