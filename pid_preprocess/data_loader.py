from dataclasses import dataclass
import json
import os
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
from dask import delayed, compute
import pandas as pd

try:
    from pid_preprocess import schemas
    from utils.file_utils import load_json_data
except ModuleNotFoundError:
    import sys
    # 프로젝트 루트 경로를 sys.path에 추가합니다.
    project_root_path = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root_path))
    print(f"[Warning] Added '{project_root_path}' to sys.path for direct execution.")
    
    from pid_preprocess import schemas
    from utils.file_utils import load_json_data

    
def normalize_and_convert_info(data_part, meta_df: pd.DataFrame):
    """pd.json_normalize와 data_created를 datatime 타입으로 변환해주고, meta 스키마에 맞게 컬럼 순서와 타입을 적용한느 함수입니다."""
    if not data_part: # 데이터가 비어있는 경우 meta와 동일한 빈 DF 반환
        return meta_df.copy()
    
    df = pd.json_normalize(data_part)
    
    # meta에 정의된 컬럼 리스트
    expected_columns = meta_df.columns
    
    # df의 컬럼을 meta의 컬럼 순서와 정확히 일치시킴
    # df에 없는 컬럼은 NaN으로 채워지고, meta에 없는 컬럼은 버려짐.
    df = df.reindex(columns=expected_columns)
    
    # 'date_created' 컬럼이 존재하면, 타입을 datetime으로 변환
    if 'date_created' in df.columns:
        df['date_created'] = pd.to_datetime(df['date_created'])
    
    # 모든 컬럼 타입을 meta와 강제로 일치시켜 안정성 확보
    df = df.astype(meta_df.dtypes)
    
    return df
    
def load_all_parts_from_file(json_file_path: Path) -> dict:
    """
    파일을 한 번만 읽고 images와 annotations 리스트를 딕셔너리로 반환합니다.
    Return dict keys:
        info: list
        images: list
        annotations: list
    """
    return load_json_data(json_file_path)


def setting_categories_data(data_path: Path, json_file_name: str="categories.json") -> pd.DataFrame:
    """
        별로도 저장된 categories 데이터를 불러오고, 필요한 정보를 추가합니다.
    """
    # 카테고리 정보 로드 및 DataFrame 형태로 변환
    categories: pd.DataFrame = pd.DataFrame(load_json_data(data_path / json_file_name))
    # 'name' 컬럼을 '@' 기준으로 분리하여 새로운 컬럼 생성
    # .str.split()에 expand=True를 사용하면 분리된 결과가 새로운 컬럼들로 만들어집니다.
    categories[["category_group", "category_name"]] = categories["name"].str.split('@', n=1, expand=True)
    # 'id'를 인덱스로 설정하여 특정 카테고리를 빠르게 찾을 수 있도록 함
    categories = categories.set_index("id")
    
    return categories


@dataclass(frozen=True) # frozen=True: 불변(immutable) 객체로 만들어 실수로 값 변경 방지
class LoadedDaskData:
    infos: dd.DataFrame
    images: dd.DataFrame
    annotations: dd.DataFrame


class DataLoader():
    def __init__(self, dataPath: Path, jsonDir: str, isLog: bool=False):
        self.json_path: Path = dataPath / jsonDir
        self.isLog = isLog
        
    def load_data(self, pattern: str) -> LoadedDaskData:
        # 파일 목록 가져오기
        file_list = sorted(self.json_path.glob(pattern=pattern))
        
        # 2. 각 파일에 대해 로더 함수를 '지연' 실행. (파일 읽기 I/O는 여기서 한 번만 계획됨)
        lazy_results = [delayed(load_all_parts_from_file)(f) for f in file_list]
        
        # 3. '지연된' 결과(딕셔너리)를 '재사용'하여 각 데이터프레임을 생성
        # 3-1. info 데이터프레임 만들기
        lazy_infos_dfs = [
            delayed(normalize_and_convert_info)(res['info'], schemas.infos_meta) 
            for res in lazy_results
            ]
        infos_ddf = dd.from_delayed(lazy_infos_dfs, meta=schemas.infos_meta)
        # 3-2. images 데이터프레임 만들기
        lazy_images_dfs = [
            delayed(normalize_and_convert_info)(res['images'], schemas.images_meta) 
            for res in lazy_results
            ]
        images_ddf = dd.from_delayed(lazy_images_dfs, meta=schemas.images_meta)

        # 3-3. annotations 데이터프레임 만들기
        lazy_annotations_dfs = [
            delayed(normalize_and_convert_info)(res['annotations'], schemas.annotations_meta) 
            for res in lazy_results
            ]
        annotations_ddf = dd.from_delayed(lazy_annotations_dfs, meta=schemas.annotations_meta)
        
        if 'bbox' in annotations_ddf.columns:
            annotations_ddf['bbox'] = annotations_ddf['bbox'].astype('object')
        
        
        if self.isLog:
            # infos, images, annotations 데이터프레임을 단 한 번의 연산으로 모두 계산
            # Dask가 전체 그래프를 최적화하여 가장 효율적으로 실행합니다.
            infos_pdf, images_pdf, annotations_pdf = compute(
                infos_ddf, images_ddf, annotations_ddf
            )
            print("--- Infos DataFrame Head ---")
            print(infos_pdf.head())

            print("--- Images DataFrame Head ---")
            print(images_pdf.head())

            print("\n--- Annotations DataFrame Head ---")
            print(annotations_pdf.head())
            print(annotations_pdf.info())
        
        return LoadedDaskData(
            infos=infos_ddf,
            images=images_ddf,
            annotations=annotations_ddf
            )


if __name__ == "__main__":
    # --- 1. 기본 경로 및 파일 경로 설정 ---
    base_dir = Path(os.getcwd()).resolve()
    
    data_path = base_dir / "assets"
    
    train_data_loader = DataLoader(dataPath=data_path, jsonDir="preprocessed_data_json", isLog=False)
    
    train_data = train_data_loader.load_data("TL_prepro/TL_*_*/*.json")
    
    print(len(train_data.infos))
    # print(len(train_data.images))
    # print(len(train_data.annotations))