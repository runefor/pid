import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
data_path = base_dir / "assets"


unique_categories = {}

for json_file in data_path.glob("TL/TL_*_*/*.json"):
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for cat in data.get("categories", []):
        cat_id = cat["id"]
        if cat_id not in unique_categories:
            unique_categories[cat_id] = cat
        else:
            # 중복인데 내용이 다르면 경고 출력
            if unique_categories[cat_id] != cat:
                print(f"⚠️ ID 충돌: {cat_id} in {json_file.name}")
    if "categories" in data:
        del data["categories"]
        with json_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"🧹 categories 제거 완료: {json_file.name}")


merged_categories = list(unique_categories.values())

output_path = data_path / "categories.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(merged_categories, f, indent=2, ensure_ascii=False)

print(f"✅ 중복 제거된 categories.json 저장 완료: {output_path.resolve()}")
