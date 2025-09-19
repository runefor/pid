import json
from pathlib import Path
from collections import defaultdict, Counter

base_dir = Path(__file__).resolve().parent
# data_path = base_dir / "../assets/roboflow_train"
# data_path = base_dir / "../assets/roboflow_valid"
data_path = base_dir / "../assets/roboflow_test"

with (data_path / "_annotations.coco.json").open("r", encoding="utf-8") as f:
    data = json.load(f)
    

print(data["annotations"][0].keys())

catecory = set()

for annotation in data["annotations"]:
    catecory.add(annotation["category_id"])
    
print(catecory)