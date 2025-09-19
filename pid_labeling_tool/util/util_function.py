import json
from pathlib import Path


def read_json(path: Path) -> list:
    try:
        content = path.read_text(encoding="utf-8")
        
        data = json.loads(content)
        
        return data
    
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        return []
    
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{path}'. Check the file format.")
        return []