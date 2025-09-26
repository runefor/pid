from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from pathlib import Path

from util.util_function import read_json

app = Flask(__name__)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
CATEGORIES_PATH = ASSETS_DIR / "categories.json"

# 카테고리 정보 미리 로드 (ID -> 이름 매핑)
id_to_name = {}
try:
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        categories_data = json.load(f)
        id_to_name = {cat["id"]: cat["name"] for cat in categories_data}
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Cannot load or parse categories.json. Category filtering will be disabled. Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/file-list')
def get_file_list():
    # 전처리된 이미지와 JSON 파일이 있는 경로를 사용합니다.
    # 어노테이션 파일을 기준으로 목록을 생성합니다.
    annotation_base_path = ASSETS_DIR / "preprocessed_data_json" / "TL_prepro"
    image_source_path = ASSETS_DIR / "TS"
    file_entries = []

    # 어노테이션 폴더(TL_prepro/TL_*_*)를 순회합니다.
    for ann_folder in annotation_base_path.glob('TL_*_*'):
        if not ann_folder.is_dir():
            continue
        
        # 각 어노테이션 파일(.json)을 기준으로 처리합니다.
        for label_path in ann_folder.glob('*.json'):
            # 해당 어노테이션에 매칭되는 이미지 경로를 생성합니다.
            # 예: TL_V01_006 -> TS_V01_006
            image_folder_name = ann_folder.name.replace('TL', 'TS')
            img_path = image_source_path / image_folder_name / label_path.with_suffix('.png').name

            # 이미지가 실제로 존재하지 않으면 목록에 추가하지 않습니다.
            if not img_path.exists():
                continue

            image_categories = set()
            # 라벨 파일에서 카테고리 정보 추출
            with open(label_path, "r", encoding="utf-8") as f:
                try:
                    label_data = json.load(f)
                    for ann in label_data.get("annotations", []):
                        cat_id = ann.get("category_id")
                        if cat_id in id_to_name:
                            image_categories.add(id_to_name[cat_id])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {label_path}")

            file_entries.append({
                'folder': ann_folder.name, # 기준이 되는 폴더는 어노테이션 폴더
                'filename': img_path.name,
                'image': str(img_path.relative_to(ASSETS_DIR)),
                'label': str(label_path.relative_to(ASSETS_DIR)) if label_path.exists() else None,
                'categories': sorted(list(image_categories)) # 추출된 카테고리 목록 추가
            })

    return jsonify(file_entries)

@app.route('/assets/<path:filename>')
def serve_asset(filename):
    # 절대 경로로 변환
    file_path = Path(ASSETS_DIR) / filename

    # 보안: ASSETS_DIR 내부에 있는 파일만 허용
    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(Path(ASSETS_DIR).resolve())):
            return "Access denied", 403
    except Exception:
        return "Invalid path", 400

    # 파일 존재 여부 확인
    if not file_path.exists():
        return "File not found", 404

    # 파일 서빙
    return send_from_directory(ASSETS_DIR, filename)

@app.route('/classes')
def get_classes():
    # 예시: 실제로는 DB나 config 파일에서 불러올 수도 있음
    datas = read_json(CATEGORIES_PATH)
    return jsonify(datas)

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    folder_name = data.get('folder') # JS에서 보낸 folder 이름
    filename = data.get('filename') # JS에서 보낸 filename

    if not folder_name or not filename:
        return jsonify({"status": "error", "message": "Folder or filename not provided"}), 400

    tl_folder = Path(ASSETS_DIR) / 'preprocessed_data_json' / 'TL_prepro' / folder_name
    tl_folder.mkdir(parents=True, exist_ok=True)

    json_path = tl_folder / f'{Path(filename).stem}.json'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved"})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
