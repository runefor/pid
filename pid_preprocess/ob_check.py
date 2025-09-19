import json
from pathlib import Path
from collections import defaultdict, Counter

base_dir = Path(__file__).resolve().parent
data_path = base_dir / "../assets"

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
annotation_counts = Counter()

for json_file in data_path.glob("TL_TL_*_*/*.json"):
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        for ann in data.get("annotations", []):
            cat_id = ann["category_id"]
            annotation_counts[cat_id] += 1

# 3. 전체 객체 수
total_objects = sum(annotation_counts.values())
print(f"🔢 전체 객체 수: {total_objects:,}")

# 4. 클래스별 비율 출력 (소분류 기준)
print("\n📊 클래스별 객체 수 및 비율:")
for cat_id, count in annotation_counts.most_common():
    name = id_to_name.get(cat_id, f"Unknown({cat_id})")
    ratio = count / total_objects * 100
    print(f"- {name:40} | {count:5}개 | {ratio:5.2f}%")

# 5. 가장 많은 클래스 TOP 5
print("\n🏆 가장 많은 클래스 TOP 5:")
for i, (cat_id, count) in enumerate(annotation_counts.most_common(5), 1):
    name = id_to_name.get(cat_id, f"Unknown({cat_id})")
    print(f"{i}. {name} → {count:,}개")

# 6. 대분류 기준 객체 수 및 비율
major_counts = defaultdict(int)
for cat_id, count in annotation_counts.items():
    major = id_to_major.get(cat_id, "Unknown")
    major_counts[major] += count

print("\n📂 대분류 기준 객체 수 및 비율:")
for major, count in sorted(major_counts.items(), key=lambda x: x[1], reverse=True):
    ratio = count / total_objects * 100
    print(f"- {major:20} | {count:6}개 | {ratio:5.2f}%")