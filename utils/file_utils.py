import json
from pathlib import Path


def load_json_data(path: Path) -> dict:
    with path.open('r', encoding="utf-8") as f:
        return json.load(f)
    
def load_multiple_json_data(paths: list[Path]) -> list[dict]:
    return [load_json_data(path) for path in paths] # 메모리 효율이 안 좋을 것 같다.
    
def save_json_data(data: dict, path: Path) -> None:
    with path.open('w', encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)