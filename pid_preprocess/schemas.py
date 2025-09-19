import pandas as pd


infos_meta = pd.DataFrame({
    # infos 관련 컬럼 및 타입 정의
    'name': pd.Series(dtype='str'),
    'url': pd.Series(dtype='str'),
    'description': pd.Series(dtype='str'),
    'version': pd.Series(dtype='str'),
    'contributor': pd.Series(dtype='str'),
    'date_created': pd.Series(dtype='datetime64[ns]'),
})

images_meta = pd.DataFrame({
    # images 관련 컬럼 및 타입 정의
    'id': pd.Series(dtype='str'),
    'file_name': pd.Series(dtype='str'),
    'width': pd.Series(dtype='int'),
    'height': pd.Series(dtype='int'),
    'date_captured': pd.Series(dtype='datetime64[ns]'),
})

annotations_meta = pd.DataFrame({
    # annotations 관련 컬럼 및 타입 정의
    'id': pd.Series(dtype='int'),
    'image_id': pd.Series(dtype='str'),
    'category_id': pd.Series(dtype='int'),
    # 'attributes': pd.Series(dtype='object'), # dict
    'attributes.pidLabel': pd.Series(dtype='str'),
    'attributes.shipType': pd.Series(dtype='str'),
    'attributes.vendor': pd.Series(dtype='str'),
    'iscrowd': pd.Series(dtype='int8'), # 0이면 단 하나의 객체, 1이면 여러 객체들
    'bbox': pd.Series(dtype='object'), # list
    'area': pd.Series(dtype='float64'),
})