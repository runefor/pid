from pathlib import Path
from typing import Union
import os

try:
    from utils.file_utils import load_json_data, save_json_data
except ModuleNotFoundError:
    import sys
    # 프로젝트 루트 경로를 sys.path에 추가합니다.
    project_root_path = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root_path))
    print(f"[Warning] Added '{project_root_path}' to sys.path for direct execution.")
    
    from utils.file_utils import load_json_data, save_json_data

def merge_per_image_jsons_to_coco(
    json_dir: Union[str, Path], 
    categories_file: Union[str, Path], 
    output_file: Union[str, Path],
    pattern: str="*.json",
):
    """
        per-image JSON들을 하나의 COCO JSON으로 병합. categories는 별도 파일에서 로드.
        - json_dir: per-image JSON 파일들이 있는 디렉토리 (e.g., image1.json, image2.json).
        - categories_file: categories.json 경로 (모든 per-image JSON의 categories와 동일 가정).
        - 각 per-image JSON 형식: {"image_id": int (옵션), "annotations": [{"bbox": [x,y,w,h], "category_id": int, ...}]}
        - image_dir: 이미지 파일 디렉토리 (width/height 추출용, 옵션).
        - output_file: 병합된 COCO JSON 저장 경로.
    """
    json_dir = Path(json_dir)
    categories_file = Path(categories_file)
    output_file = Path(output_file)
    
    # categories 로드
    categories = load_json_data(categories_file)
    
    merged = {
        "info": {
            "name": "조선·해양 플랜트 P&ID 심볼 식별 데이터",
            "url": "miraeit.net/",
            "description": "조선·해양 플랜트 P&ID 심볼 식별 데이터",
            "version": "1.0",
            "contributor": "디에스엠이정보시스템컨소시엄",
            "date_created": "2022-12-02"
        },
        "licenses": [
            {
            "id": 1,
            "name": "CC BY-NC",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories  # 별도 파일에서 직접 사용
    }
    
    file_list = sorted(json_dir.glob(pattern=pattern))

    for json_file in sorted(file_list):
        data = load_json_data(json_file)
        
        merged["images"].extend(data.get("images", {}))
        
        merged["annotations"].extend(data.get("annotations", {}))


    # JSON 저장
    save_json_data(merged, output_file)
    print(f"Merged COCO JSON saved to {output_file}")
    print(merged["images"][:2])
    print(merged["annotations"][:2])


if __name__ == "__main__":
    base_dir = Path(os.getcwd()).resolve()
    
    data_path = base_dir / "assets"
    
    merge_per_image_jsons_to_coco(
        json_dir=data_path / "preprocessed_data_json/VL_prepro",
        categories_file=data_path / "categories.json",
        output_file=data_path / "merged_VL_prepro.json",
        pattern="VL_*_*/*.json"
    )