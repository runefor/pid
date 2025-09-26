import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    from utils.tile_utils import tile_image
except ImportError:
    # 상대 경로로 utils 디렉토리를 추가
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.tile_utils import tile_image

def visualize_tiling(original_image_path=None, tile_size=640, overlap=0.5, save_path='tiling_cv2.png'):
    """
    수정된 cv2 타일링 시각화: 패딩 회색, 텍스트 선명, 오버레이 추가.
    """
    # 이미지 로드 (더미 생성)
    if original_image_path is None:
        h, w = 2048, 2048
        image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        print("더미 이미지 생성: 2048x2048")
    else:
        image = cv2.imread(original_image_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {original_image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"이미지 로드: {original_image_path}")
    
    # 타일링
    tiles = tile_image(image, tile_size=tile_size, overlap=overlap)
    num_tiles = len(tiles)
    print(f"타일 수: {num_tiles} (stride: {int(tile_size * (1 - overlap))})")
    
    # 그리드 계산
    cols = int(np.ceil(np.sqrt(num_tiles)))
    rows = int(np.ceil(num_tiles / cols))
    
    # 큰 캔버스 생성
    canvas_height = rows * tile_size
    canvas_width = cols * tile_size
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 128  # 회색 배경 (패딩 대비)
    
    # 타일 배치 + 경계/텍스트
    for idx, (tile, offset) in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        y_start = row * tile_size
        x_start = col * tile_size
        
        # 타일 복사
        canvas[y_start:y_start+tile_size, x_start:x_start+tile_size] = tile
        
        # 빨간 경계 (두껍게)
        cv2.rectangle(canvas, (x_start, y_start), (x_start+tile_size, y_start+tile_size), (0, 0, 255), 3)
        
        # 오프셋 텍스트 (선명: 흰색 + 검은 테두리)
        text = f"T{idx}: ({offset[0]},{offset[1]})"
        # 검은 테두리 (오프셋으로 여러 번 그리기)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cv2.putText(canvas, text, (x_start + 10 + dx, y_start + 30 + dy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        # 흰색 메인
        cv2.putText(canvas, text, (x_start + 10, y_start + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 저장 (BGR로 변환)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, canvas_bgr)
    print(f"캔버스 저장: {save_path} (크기: {canvas_width}x{canvas_height})")

def overlay_tiles_on_original(image: np.ndarray, tiles: list, tile_size: int, save_path='overlay_on_original.png'):
    """
    추가: 원본 이미지에 타일 경계 오버레이 (분할 확인에 좋음).
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    
    stride = int(tile_size * (1 - 0.2))  # overlap=0.2 가정
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # 타일 경계 그리기 (빨간색, 투명도)
            end_y = min(y + tile_size, h)
            end_x = min(x + tile_size, w)
            cv2.rectangle(overlay, (x, y), (end_x, end_y), (0, 0, 255), 2)
            # 오프셋 텍스트
            cv2.putText(overlay, f"({x},{y})", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay_bgr)
    print(f"오버레이 저장: {save_path}")

if __name__ == "__main__":

    base_dir = Path(os.getcwd()).resolve()
    
    data_path = base_dir / "assets"
    
    image_path = data_path / "image" / "all" / "images" / "V01_03_016_001_1.png"
    
    # visualize_tiling(save_fig_path=str(image_path.with_name(f"{image_path.stem}_tiled.png")))
    visualize_tiling(original_image_path=str(image_path))
