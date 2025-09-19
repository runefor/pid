import json
from collections import defaultdict
from pathlib import Path

base_dir = Path(__file__).resolve().parent
# categories.json 경로
categories_path = base_dir / Path("../assets/categories.json")

# 결과 저장용 딕셔너리
category_count = defaultdict(set)

# JSON 로딩
with categories_path.open("r", encoding="utf-8") as f:
    categories = json.load(f)

# 대분류별 소분류 수집
for cat in categories:
    name = cat.get("name", "")
    if "@" in name:
        major, minor = name.split("@", 1)
        category_count[major].add(minor)

# 결과 출력
for major, minors in category_count.items():
    print(f"{major}: {len(minors)}개 소분류")