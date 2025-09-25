# main_preprocess_pipeline.py
import dataclasses
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from pid_preprocess.data_loader import DataLoader, setting_categories_data
from pid_preprocess.feature_engineering import add_bbox_features
from pid_preprocess import eda_runner
from utils.file_utils import load_json_data


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

def test_visualize_class_distribution():
    """
    merged_dataset.json 파일의 클래스 분포를 분석하고 시각화합니다.
    """
    print("📊 Starting class distribution visualization...")

    # --- 1. 경로 설정 ---
    base_dir = Path(os.getcwd()).resolve()
    data_path = base_dir / "assets"
    report_path = base_dir / "reports"
    figure_path = report_path / "figures"

    # 출력 디렉토리 생성
    figure_path.mkdir(parents=True, exist_ok=True)

    # --- 2. 데이터 로드 ---
    json_file_path = data_path / "merged_dataset.json"
    if not json_file_path.exists():
        print(f"❌ Error: File not found at {json_file_path}")
        return

    print(f"   Loading data from {json_file_path}...")
    coco_data = load_json_data(json_file_path)
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    if not annotations or not categories:
        print("   ⚠️ Annotations or categories are missing in the JSON file.")
        return

    # --- 3. 데이터 분석 ---
    # 카테고리 ID와 이름 매핑
    id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # 어노테이션에서 카테고리 ID 카운트
    class_counts = Counter(ann["category_id"] for ann in annotations)

    # 카운트된 ID를 이름으로 변환하고, 개수 순으로 정렬
    class_distribution = sorted(
        [(id_to_name.get(cat_id, f"Unknown({cat_id})"), count) for cat_id, count in class_counts.items()],
        key=lambda item: item[1],
        reverse=True
    )

    # --- 4. 시각화 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # fig, ax 객체를 받아와서 세부적인 설정을 합니다.
    fig, ax = plt.subplots(figsize=(14, 24)) # 가로 길이를 늘려 숫자 표시 공간 확보

    class_names, counts = zip(*class_distribution)
    sns.barplot(x=list(counts), y=list(class_names), orient='h', ax=ax)

    # 각 막대(bar)에 정확한 수치를 텍스트로 추가합니다.
    # ax.containers[0]는 barplot이 생성한 막대 그룹을 의미합니다.
    # fmt='{:,.0f}'는 숫자를 천 단위 콤마로 포맷팅합니다.
    ax.bar_label(ax.containers[0], fmt='{:,.0f}', padding=5, fontsize=10)

    ax.set_title('Class Distribution in merged_dataset.json', fontsize=16)
    ax.set_xlabel('Number of Annotations', fontsize=12)
    ax.set_ylabel('Class Name', fontsize=12)
    ax.set_xlim(right=max(counts) * 1.15) # 숫자 레이블이 잘리지 않도록 x축 범위 확장
    plt.tight_layout()

    # --- 5. 결과 저장 ---
    save_file = figure_path / "merged_dataset_class_distribution.png"
    plt.savefig(save_file)
    print(f"   ✅ Visualization saved to: {save_file}")

def test_visualize_rare_class_locality():
    """
    어노테이션이 100개 이하인 소수 클래스들이
    소수의 이미지에 집중되어 있는지, 여러 이미지에 흩어져 있는지 분석하고 시각화합니다.
    """
    print("🔬 Starting rare class locality visualization...")

    # --- 1. 경로 및 파라미터 설정 ---
    base_dir = Path(os.getcwd()).resolve()
    data_path = base_dir / "assets"
    figure_path = base_dir / "reports" / "figures"
    figure_path.mkdir(parents=True, exist_ok=True)

    rare_threshold = 100

    # --- 2. 데이터 로드 ---
    json_file_path = data_path / "merged_dataset.json"
    if not json_file_path.exists():
        print(f"❌ Error: File not found at {json_file_path}")
        return

    print(f"   Loading data from {json_file_path}...")
    coco_data = load_json_data(json_file_path)
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])

    if not annotations or not categories:
        print("   ⚠️ Annotations or categories are missing in the JSON file.")
        return

    # --- 3. 데이터 분석 ---
    id_to_name = {cat["id"]: cat["name"] for cat in categories}

    # 전체 클래스별 어노테이션 수 계산
    total_class_counts = Counter(ann["category_id"] for ann in annotations)

    # 소수 클래스(rare class) ID 필터링
    rare_class_ids = {
        cat_id for cat_id, count in total_class_counts.items() if count <= rare_threshold
    }

    # 소수 클래스가 등장하는 이미지 ID 수집
    class_to_images = defaultdict(set)
    for ann in annotations:
        if ann["category_id"] in rare_class_ids:
            class_to_images[ann["category_id"]].add(ann["image_id"])

    # 시각화를 위한 데이터 구조 생성: (클래스명, 총 어노테이션 수, 고유 이미지 수)
    locality_data = []
    for cat_id in rare_class_ids:
        class_name = id_to_name.get(cat_id, f"Unknown({cat_id})")
        total_annotations = total_class_counts[cat_id]
        unique_image_count = len(class_to_images[cat_id])
        locality_data.append((class_name, total_annotations, unique_image_count))

    # 고유 이미지 수, 그 다음 총 어노테이션 수로 정렬
    locality_data.sort(key=lambda x: (x[2], x[1]), reverse=True)

    if not locality_data:
        print(f"   ℹ️ No classes found with {rare_threshold} or fewer annotations.")
        return

    # --- 4. 시각화 ---
    class_names, total_counts, image_counts = zip(*locality_data)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, max(10, len(class_names) * 0.4))) # 클래스 수에 따라 높이 조절

    # 막대 그래프: 막대 길이는 '고유 이미지 수'
    sns.barplot(x=list(image_counts), y=list(class_names), orient='h', ax=ax, color='skyblue')

    # 막대 끝에 '총 어노테이션 수'를 텍스트로 표시
    ax.bar_label(ax.containers[0], labels=[f'{c:,}' for c in total_counts], padding=5, fontsize=10, color='black')

    ax.set_title(f'Locality of Rare Classes (<= {rare_threshold} annotations)', fontsize=16)
    ax.set_xlabel('Number of Unique Images (Bar Length) & Total Annotations (Label)', fontsize=12)
    ax.set_ylabel('Class Name', fontsize=12)
    ax.set_xlim(right=max(total_counts) * 1.2) # 텍스트가 잘리지 않도록 x축 범위 확장
    plt.tight_layout()

    # --- 5. 결과 저장 ---
    save_file = figure_path / "rare_class_locality.png"
    plt.savefig(save_file)
    print(f"   ✅ Visualization saved to: {save_file}")

def test_report_rare_class_image_distribution():
    """
    100개 이하의 어노테이션을 가진 소수 클래스 각각이
    어떤 이미지에 몇 개의 어노테이션으로 존재하는지 상세 리포트를 생성합니다.
    """
    print("📄 Generating rare class image distribution report...")

    # --- 1. 경로 및 파라미터 설정 ---
    base_dir = Path(os.getcwd()).resolve()
    data_path = base_dir / "assets"
    report_path = base_dir / "reports"
    report_path.mkdir(parents=True, exist_ok=True)

    rare_threshold = 100

    # --- 2. 데이터 로드 ---
    json_file_path = data_path / "merged_dataset.json"
    if not json_file_path.exists():
        print(f"❌ Error: File not found at {json_file_path}")
        return

    print(f"   Loading data from {json_file_path}...")
    coco_data = load_json_data(json_file_path)
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    images = coco_data.get("images", [])

    if not all([annotations, categories, images]):
        print("   ⚠️ Annotations, categories, or images are missing in the JSON file.")
        return

    # --- 3. 데이터 분석 ---
    # ID-이름 매핑 생성
    id_to_cat_name = {cat["id"]: cat["name"] for cat in categories}
    id_to_img_name = {img["id"]: img["file_name"] for img in images}

    # 전체 클래스별 어노테이션 수 계산
    total_class_counts = Counter(ann["category_id"] for ann in annotations)

    # 소수 클래스(rare class) ID 필터링
    rare_class_ids = {
        cat_id for cat_id, count in total_class_counts.items() if count <= rare_threshold
    }

    # 소수 클래스에 대해 {클래스 ID: {이미지 ID: 개수}} 형태로 집계
    class_image_counts = defaultdict(lambda: defaultdict(int))
    for ann in annotations:
        cat_id = ann["category_id"]
        if cat_id in rare_class_ids:
            class_image_counts[cat_id][ann["image_id"]] += 1

    # --- 4. 리포트 생성 ---
    report_lines = []
    
    # 클래스 이름순으로 정렬하여 리포트 일관성 유지
    sorted_rare_class_ids = sorted(list(rare_class_ids), key=lambda cid: id_to_cat_name.get(cid, ''))

    for cat_id in sorted_rare_class_ids:
        class_name = id_to_cat_name.get(cat_id, f"Unknown({cat_id})")
        total_count = total_class_counts[cat_id]
        
        report_lines.append(f"Class: {class_name} (Total Annotations: {total_count})")
        
        # 이미지별 분포: 이미지 내 어노테이션 개수 순으로 정렬
        image_distribution = sorted(
            class_image_counts[cat_id].items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        
        for img_id, count in image_distribution:
            img_name = id_to_img_name.get(img_id, f"Unknown Image ID({img_id})")
            report_lines.append(f"  - Image: {img_name}, Annotations: {count}")
        report_lines.append("-" * 40)

    # --- 5. 결과 저장 ---
    save_file = report_path / "rare_class_image_distribution.txt"
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"   ✅ Report saved to: {save_file}")


if __name__ == "__main__":
    # data_pipeline()
    # test_visualize_class_distribution()
    # test_visualize_rare_class_locality()
    test_report_rare_class_image_distribution()