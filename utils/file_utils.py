import json
from pathlib import Path


def load_json_data(path: Path) -> dict:
    with path.open('r', encoding="utf-8") as f:
        return json.load(f)
    
def save_json_data(data: dict, path: Path) -> None:
    with path.open('w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)