import ast
import pandas as pd
import dask.dataframe as dd


def expand_bbox_with_coords(bbox_data: list[float]):
    if isinstance(bbox_data, str):
        # 만약 데이터가 문자열이면, ast.literal_eval로 실제 리스트로 변환
        bbox_list = ast.literal_eval(bbox_data)
    else:
        # 이미 리스트이면 그대로 사용
        bbox_list = bbox_data
        
    x, y, w, h = bbox_list
    x2 = x + w
    y2 = y + h
    return pd.Series([x, y, w, h, x2, y2], index=['x', 'y', 'w', 'h', 'x2', 'y2'])

def cal_bbox_info(bbox_data: list[float]):
    if isinstance(bbox_data, str):
        bbox_list = ast.literal_eval(bbox_data)
    else:
        bbox_list = bbox_data
    
    x, y, w, h = bbox_list
    aspect_ratio = w / (h if h != 0 else 1e-6)
    center_x = (x + w) / 2
    center_y = (y + h) / 2
    return pd.Series([aspect_ratio, center_x, center_y], index=['aspect_ratio', 'center_x', 'center_y'])

def add_bbox_features(annotations_ddf: dd.DataFrame) -> dd.DataFrame:
    """
    annotations 데이터프레임에 bbox 관련 피처들을 추가하는 파이프라인 함수
    """
    print("Adding bounding box features...")
    
    # 1. 좌표 확장
    coord_meta = {'x': 'f8', 'y': 'f8', 'w': 'f8', 'h': 'f8', 'x2': 'f8', 'y2': 'f8'}
    new_cols_coords = annotations_ddf['bbox'].apply(expand_bbox_with_coords, meta=coord_meta)
    
    # 2. 추가 정보 계산
    info_meta = {'aspect_ratio': 'f8', 'center_x': 'f8', 'center_y': 'f8'}
    new_cols_info = annotations_ddf['bbox'].apply(cal_bbox_info, meta=info_meta)
    
    # 3. 원본 데이터프레임과 병합
    return dd.concat([annotations_ddf, new_cols_coords, new_cols_info], axis=1)