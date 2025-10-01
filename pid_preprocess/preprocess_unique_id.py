from pathlib import Path
import json
import pandas as pd
import dask
from dask import delayed
from dask.diagnostics import ProgressBar
from tqdm import tqdm

def load_and_optimize_with_unique_id(json_file_path):
    """
    (이전과 동일) JSON 파일을 읽고 고유 ID를 생성/적용한 뒤,
    최적화된 images와 annotations Pandas DataFrame을 딕셔너리로 반환합니다.
    """
    try:
        unique_id = Path(json_file_path).stem
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        images_data = data.get('images', [])
        if images_data:
            for item in images_data: item['id'] = unique_id
            images_df = pd.json_normalize(images_data)
            images_df['id'] = images_df['id'].astype(str)
            if 'width' in images_df.columns: images_df['width'] = pd.to_numeric(images_df['width'], downcast='unsigned')
            if 'height' in images_df.columns: images_df['height'] = pd.to_numeric(images_df['height'], downcast='unsigned')
        else:
            images_df = pd.DataFrame()

        annotations_data = data.get('annotations', [])
        if annotations_data:
            for item in annotations_data: item['image_id'] = unique_id
            annotations_df = pd.json_normalize(annotations_data)
        else:
            annotations_df = pd.DataFrame()
            
        return {'images': images_df, 'annotations': annotations_df}
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
        return {'images': pd.DataFrame(), 'annotations': pd.DataFrame()}

def process_and_save_single_file(json_path, input_root, output_root):
    """
    하나의 JSON 파일을 받아 처리하고, 지정된 경로에 Parquet 파일로 저장합니다.
    """
    # 1. 파일 읽기 및 데이터프레임 생성
    dataframes = load_and_optimize_with_unique_id(json_path)
    images_df = dataframes['images']
    annotations_df = dataframes['annotations']

    # 처리할 데이터가 없으면 종료
    if images_df.empty and annotations_df.empty:
        return f"Skipped empty/error file: {json_path.name}"

    # 2. 원본 구조를 유지하는 출력 경로 생성
    relative_path = json_path.relative_to(input_root)
    
    # 상위 폴더 경로 변경 (예: 'TL' -> 'TL_prepro')
    path_parts = list(relative_path.parts)
    if path_parts and path_parts[0] == 'TL':
        path_parts[0] = 'TL_prepro'
    
    # 출력 디렉토리 경로 조합 및 생성
    output_dir = output_root.joinpath(*path_parts[:-1])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. 각 DataFrame을 별도의 Parquet 파일로 저장
    file_stem = json_path.stem
    if not images_df.empty:
        images_df.to_parquet(output_dir / f"{file_stem}_images.parquet", engine='pyarrow')
    if not annotations_df.empty:
        annotations_df.to_parquet(output_dir / f"{file_stem}_annotations.parquet", engine='pyarrow')
    
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
        delayed(process_and_save_single_file)(f, input_path, output_path) 
        for f in file_list
    ]
    
    # Dask Progress Bar와 함께 모든 지연 작업 병렬 실행
    with ProgressBar():
        print("병렬 처리 및 저장 시작...")
        results = dask.compute(*lazy_tasks)
        print("병렬 처리 및 저장 완료.")
    
    # 결과 요약 (선택 사항)
    # for res in results[0][:10]: # 처음 10개 결과만 출력
    #     print(res)


# --- 스크립트 실행 ---
if __name__ == "__main__":
    # 원본 데이터가 있는 상위 경로
    base_dir = Path(__file__).resolve().parent.parent
    INPUT_ROOT_DIR = base_dir / "assets"
    
    # 전처리된 Parquet 파일을 저장할 경로
    OUTPUT_ROOT_DIR = INPUT_ROOT_DIR / 'preprocessed_data'
    
    # JSON 파일 검색 패턴
    # 예: 'TL/TL_V01_006/*.json' -> TL/TL_V01_006 폴더 안의 json만
    # 'TL/**/*.json' -> TL 폴더 아래의 모든 하위 폴더 검색
    JSON_PATTERN = "TL/**/*.json"

    main_process(INPUT_ROOT_DIR, OUTPUT_ROOT_DIR, JSON_PATTERN)