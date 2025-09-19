from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from pathlib import Path

from util.util_function import read_json

app = Flask(__name__)

# 경로 설정
ASSETS_DIR = os.path.abspath(os.path.join(app.root_path, '..', 'assets')) # TODO: 여기 Path를 사용하도록 변경해야함!!

@app.route('/')
def index():
    ts_root = Path(ASSETS_DIR) / 'TS'
    tl_root = Path(ASSETS_DIR) / 'TL'

    folder_images = {}

    for ts_folder in ts_root.glob('TS_V*_*'):
        if not ts_folder.is_dir():
            continue

        folder_name = ts_folder.name
        tl_folder_name = folder_name.replace('TS', 'TL')
        tl_folder = tl_root / tl_folder_name

        image_files = []
        for img_path in ts_folder.glob('*.png'):
            image_name = img_path.name
            label_path = tl_folder / f'{image_name.split(".")[0]}.json'
            image_files.append({
                'image': f'TS/{folder_name}/{image_name}',
                'label': str(label_path.relative_to(ASSETS_DIR)) if label_path.exists() else None
            })

        folder_images[folder_name] = image_files

    return render_template('index.html', folder_images=folder_images)

@app.route('/file-list')
def get_file_list():
    ts_root = Path(ASSETS_DIR) / 'TS'
    file_entries = []

    for folder in ts_root.glob('TS_V*_*'):
        if not folder.is_dir():
            continue
        for img_path in folder.glob('*.png'):
            label_folder = Path(ASSETS_DIR) / 'TL' / folder.name.replace('TS', 'TL')
            label_path = label_folder / f'{img_path.stem}.json'
            file_entries.append({
                'folder': folder.name,
                'filename': img_path.name,
                'image': f'TS/{folder.name}/{img_path.name}',
                'label': str(label_path.relative_to(ASSETS_DIR)) if label_path.exists() else None
            })

    return jsonify(file_entries)

@app.route('/images/<folder>')
def get_images(folder):
    ts_folder = Path(ASSETS_DIR) / 'TS' / folder
    tl_folder = Path(ASSETS_DIR) / 'TL' / folder.replace('TS', 'TL')

    image_data = []
    for img_path in ts_folder.glob('*.png'):
        label_path = tl_folder / f'{img_path.stem}.json'
        image_data.append({
            'image': f'TS/{folder}/{img_path.name}',
            'label': str(label_path.relative_to(ASSETS_DIR)) if label_path.exists() else None
        })

    return jsonify(image_data)

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
    datas = read_json(Path(f'{ASSETS_DIR}/categories.json'))
    return jsonify(datas)

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json()
    image_filename = data.get('images', [{}])[0].get('file_name')
    folder_name = image_filename.split('_')[0] + '_' + image_filename.split('_')[1]  # 예: V01_01
    tl_folder = Path(ASSETS_DIR) / 'TL' / f'TL_{folder_name}'
    tl_folder.mkdir(parents=True, exist_ok=True)

    json_path = tl_folder / f'{Path(image_filename).stem}.json'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved"})

if __name__ == '__main__':
    app.run(debug=True)
