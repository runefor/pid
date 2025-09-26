import json
import os
from pathlib import Path
from collections import defaultdict, Counter

try:
    from utils.file_utils import load_json_data
except ModuleNotFoundError:
    import sys
    # 프로젝트 루트 경로를 sys.path에 추가합니다.
    project_root_path = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root_path))
    print(f"[Warning] Added '{project_root_path}' to sys.path for direct execution.")
    
    from utils.file_utils import load_json_data

base_dir = Path(os.getcwd()).resolve()
data_path = base_dir / "assets"
json_root_path = data_path / "preprocessed_data_json"

# 결과를 저장할 마크다운 파일 경로
output_md_path = data_path / "class_distribution_report.md"

# 1. categories.json 로드
with (data_path / "categories.json").open("r", encoding="utf-8") as f:
    categories = json.load(f)

# category_id → name 매핑
id_to_name = {cat["id"]: cat["name"] for cat in categories}

# category_id → 대분류 매핑
id_to_major = {}
for cat in categories:
    name = cat.get("name", "")
    if "@" in name:
        major = name.split("@")[0]
    else:
        major = "Unknown"
    id_to_major[cat["id"]] = major

# 2. 객체 수 집계

def class_counts(json_paths: list[Path]) -> tuple[Counter, dict]:
    annotation_counts = Counter()
    class_to_images = defaultdict(set)

    for json_file in json_paths:
        data = load_json_data(json_file)
        # 각 어노테이션에서 직접 image_id를 가져와야 합니다.
        # 이렇게 하면 병합된 파일과 개별 파일 모두에서 올바르게 동작합니다.
        for ann in data.get("annotations", []):
            cat_id = ann["category_id"]
            image_id = ann["image_id"]
            annotation_counts[cat_id] += 1
            class_to_images[cat_id].add(image_id)
    return annotation_counts, class_to_images


json_paths = list(json_root_path.glob("TL_prepro/TL_*_*/*.json"))
json_path = [data_path / "merged_dataset.json"]

annotation_counts, class_to_images = class_counts(json_paths=json_paths)
# annotation_counts, class_to_images = class_counts(json_paths=json_path)

# 3. 마크다운 리포트 생성을 위한 리스트 초기화
report_lines = ["# 📊 데이터셋 객체 분포 분석 리포트\n"]

# 4. 전체 객체 수
total_objects = sum(annotation_counts.values())
report_lines.append(f"## 🔢 전체 객체 수: **{total_objects:,}** 개\n")

# 5. 대분류 기준 객체 수 및 비율
major_counts = defaultdict(int)
for cat_id, count in annotation_counts.items():
    major = id_to_major.get(cat_id, "Unknown")
    major_counts[major] += count

report_lines.append("## 📂 대분류 기준 객체 수 및 비율\n")
report_lines.append("| 대분류 | 객체 수 | 비율 (%) |")
report_lines.append("|:---|---:|---:|")
for major, count in sorted(major_counts.items(), key=lambda x: x[1], reverse=True):
    ratio = count / total_objects * 100
    report_lines.append(f"| {major} | {count:,} | {ratio:.2f}% |")
report_lines.append("\n")

# 6. 클래스 지역성 분석 (집중도)
locality_data = []
for cat_id, total_anns in annotation_counts.items():
    name = id_to_name.get(cat_id, f"Unknown({cat_id})")
    unique_images = len(class_to_images.get(cat_id, set()))
    avg_per_image = total_anns / unique_images if unique_images > 0 else 0
    locality_data.append((name, total_anns, unique_images, avg_per_image))

# 이미지 당 평균 객체 수가 높은 순 (집중도가 높은 순)으로 정렬
locality_data.sort(key=lambda x: x[3], reverse=True)

report_lines.append("## 🔬 클래스 지역성 분석 (집중도 순)\n")
report_lines.append("`이미지 당 평균 객체 수`가 높을수록 소수의 이미지에 객체들이 집중되어 있음을 의미합니다.\n")
report_lines.append("| 소분류 (카테고리명) | 총 어노테이션 수 | 고유 이미지 수 | 이미지 당 평균 객체 수 |")
report_lines.append("|:---|---:|---:|---:|")
for name, total_anns, unique_images, avg_per_image in locality_data:
    report_lines.append(f"| {name} | {total_anns:,} | {unique_images:,} | **{avg_per_image:.2f}** |")
report_lines.append("\n")


# 6. 클래스별 비율 출력 (소분류 기준)
report_lines.append("## 📊 소분류 기준 객체 수 및 비율 (전체)\n")
report_lines.append("| 소분류 (카테고리명) | 객체 수 | 비율 (%) |")
report_lines.append("|:---|---:|---:|")
for cat_id, count in annotation_counts.most_common():
    name = id_to_name.get(cat_id, f"Unknown({cat_id})")
    ratio = count / total_objects * 100
    report_lines.append(f"| {name} | {count:,} | {ratio:.2f}% |")
report_lines.append("\n")

# 7. 전체 소분류 목록 (대분류별)
major_to_minors = defaultdict(list)
for cat in categories:
    name = cat.get("name", "")
    if "@" in name:
        major, minor = name.split("@", 1)
        major_to_minors[major].append(minor)
    else:
        major_to_minors["Unknown"].append(name)

report_lines.append("## 📋 전체 소분류 목록 (대분류별)\n")
for major, minors in sorted(major_to_minors.items()):
    report_lines.append(f"### {major} ({len(minors)}개)")
    # 2열로 표시하기 위해 리스트를 반으로 나눔
    mid_point = (len(minors) + 1) // 2
    sorted_minors = sorted(minors)
    col1 = sorted_minors[:mid_point]
    col2 = sorted_minors[mid_point:]
    
    # 각 열의 길이를 맞춤
    while len(col1) < len(col2):
        col1.append("")

    for i in range(len(col2)):
        report_lines.append(f"- {col1[i]:<40} - {col2[i]}")
    report_lines.append("\n")

# 8. 마크다운 파일로 저장
with open(output_md_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"✅ 분석 리포트가 '{output_md_path}'에 저장되었습니다.")