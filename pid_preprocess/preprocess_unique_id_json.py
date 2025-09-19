from pathlib import Path
import json
import dask
from dask import delayed
from dask.diagnostics import ProgressBar

# tqdm은 Dask의 ProgressBar와 함께 사용할 필요가 없으므로 import에서 제거했습니다.
# pandas도 더 이상 필요하지 않아 제거했습니다.

def load_and_modify_json(json_file_path):
    """
    JSON 파일을 읽고, 파일명을 기반으로 고유 ID를 생성하여
    'images'와 'annotations' 섹션에 적용한 뒤, 수정된 데이터 전체(dict)를 반환합니다.
    """
    try:
        unique_id = Path(json_file_path).stem
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 'images' 리스트가 존재하면 각 항목의 'id'를 unique_id로 변경
        if 'images' in data and data['images']:
            for item in data['images']:
                item['id'] = unique_id
        
        # 'annotations' 리스트가 존재하면 각 항목의 'image_id'를 unique_id로 변경
        if 'annotations' in data and data['annotations']:
            for item in data['annotations']:
                item['image_id'] = unique_id
                
        return data
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        # 오류 발생 시 None을 반환하여 후속 처리에서 건너뛰도록 함
        return None

def process_and_save_json(json_path, input_root, output_root):
    """
    하나의 JSON 파일을 받아 처리하고, 지정된 경로에 원본과 같은 JSON 형식으로 저장합니다.
    """
    # 1. 파일 읽기 및 데이터 수정
    modified_data = load_and_modify_json(json_path)

    # 처리 중 오류가 발생했거나 데이터가 비어있으면 종료
    if not modified_data:
        return f"Skipped empty/error file: {json_path.name}"

    # 2. 원본 구조를 유지하는 출력 경로 생성 (기존과 동일)
    relative_path = json_path.relative_to(input_root)
    
    # 상위 폴더 경로 변경 (예: 'TL' -> 'TL_prepro')
    path_parts = list(relative_path.parts)
    # if path_parts and path_parts[0] == 'TL':
    #     path_parts[0] = 'TL_prepro'
    if path_parts and path_parts[0] == 'VL':
        path_parts[0] = 'VL_prepro'
    
    # 출력 디렉토리 경로 조합 및 생성
    output_dir = output_root.joinpath(*path_parts[:-1])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 수정된 데이터를 새로운 JSON 파일로 저장
    # 파일 이름은 원본과 동일하게 유지
    output_json_path = output_dir / json_path.name
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False: 한글 깨짐 방지
        # indent=4: 가독성을 위해 들여쓰기 적용
        json.dump(modified_data, f, ensure_ascii=False, indent=4)
    
    return f"Processed: {relative_path}"


def main_process(input_dir, output_dir, file_pattern):
    """
    전체 프로세스를 관리하는 메인 함수
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 모든 JSON 파일 목록 검색
    file_list = list(input_path.glob(file_pattern))
    if not file_list:
        print("처리할 파일을 찾을 수 없습니다.")
        return
    print(f"총 {len(file_list)}개의 파일을 병렬로 처리합니다.")

    # 각 파일에 대한 처리 및 저장 작업을 '지연' 작업으로 등록
    lazy_tasks = [
        delayed(process_and_save_json)(f, input_path, output_path) 
        for f in file_list
    ]
    
    # Dask Progress Bar와 함께 모든 지연 작업 병렬 실행
    with ProgressBar():
        print("병렬 처리 및 저장 시작...")
        # dask.compute는 튜플을 반환하므로, 첫 번째 요소를 results로 받습니다.
        results = dask.compute(*lazy_tasks)
        print("병렬 처리 및 저장 완료.")
    
    # 결과 요약 (선택 사항)
    # print("\n--- 처리 결과 (상위 10개) ---")
    # for res in results[:10]:
    #     print(res)


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 원본 데이터가 있는 상위 경로
    # __file__은 스크립트 파일의 경로를 나타냅니다.
    base_dir = Path(__file__).resolve().parent
    INPUT_ROOT_DIR = base_dir / "../assets"
    
    # 전처리된 JSON 파일을 저장할 경로
    OUTPUT_ROOT_DIR = './preprocessed_data_json' # 경로 이름 변경
    
    # JSON 파일 검색 패턴
    # 예: 'TL/TL_V01_006/*.json' -> TL/TL_V01_006 폴더 안의 json만
    # 'TL/**/*.json' -> TL 폴더 아래의 모든 하위 폴더 검색
    # JSON_PATTERN = "TL/**/*.json"
    JSON_PATTERN = "VL/**/*.json"

    main_process(INPUT_ROOT_DIR, OUTPUT_ROOT_DIR, JSON_PATTERN)