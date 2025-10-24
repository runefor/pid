import argparse
from pathlib import Path
from PIL import Image, ImageDraw

try:
    from utils.tile_utils import generate_tiles
except ModuleNotFoundError:
    import sys
    # 프로젝트 루트 경로를 sys.path에 추가합니다.
    project_root_path = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root_path))
    print(f"[Warning] Added '{project_root_path}' to sys.path for direct execution.")
    
    from utils.tile_utils import generate_tiles

def create_visualization_guide(output_dir: Path, overview_image_name: str):
    """Creates a markdown file to guide the user on how to verify the tiling results."""
    md_content = f"""
# 타일링 시각화 결과 가이드

아래 이미지는 원본(왼쪽)과 타일링 영역이 표시된 개요(오른쪽)를 나란히 보여줍니다.

![Tiling Overview]({overview_image_name})

---

## 성공적인 타일링 확인 방법

아래 항목들을 확인하여 타일링이 의도대로 잘 수행되었는지 검증할 수 있습니다.

### 1. 전체 영역 커버 (Full Coverage)
- **확인 사항**: 오른쪽 '타일링 개요' 이미지에서 빨간색 사각형들이 원본 이미지의 모든 영역을 빠짐없이 덮고 있는지 확인합니다.
- **포인트**: 특히 이미지의 가장자리와 네 코너 부분이 타일에 잘 포함되었는지 확인하는 것이 중요합니다. 비는 공간이 없어야 합니다.

### 2. 적절한 겹침 (Sufficient Overlap)
- **확인 사항**: 인접한 빨간색 사각형들이 서로 적절히 겹쳐져 있는지 확인합니다.
- **포인트**: 겹치는 영역이 너무 적거나 없으면, 경계선에 걸친 객체가 분리되어 탐지 성능이 저하될 수 있습니다. 스크립트 실행 시 `--overlap` 인자로 지정한 비율(e.g., 0.2는 20%)만큼 일관되게 겹쳐 보이면 성공입니다.

### 3. 타일 크기 확인 (Tile Size Check)
- **확인 사항**: 결과 폴더에 생성된 개별 타일 이미지들(`tile_*.png`)의 크기를 확인합니다.
- **포인트**: 대부분의 타일은 `--tile_size`로 지정한 크기(e.g., 640x640)와 일치해야 합니다. 단, 원본 이미지의 오른쪽과 아래쪽 가장자리에 위치한 타일들은 전체 이미지 크기에 맞춰지므로 더 작을 수 있으며, 이는 정상적인 동작입니다.
"""
    md_path = output_dir / "tiling_visualization_guide.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content.strip())
    print(f"Saved visualization guide to {md_path}")

def visualize_tiling(image_path: Path, output_dir: Path, tile_size: int, overlap: float):
    """
    Loads an image, generates tiles, saves each tile, and creates a side-by-side comparison image.
    """
    # 1. Setup
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        img = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return

    img_w, img_h = img.size
    print(f"Loaded image: {image_path.name} (Width: {img_w}, Height: {img_h})")

    # Create a copy for drawing the overview
    overview_img = img.copy()
    draw = ImageDraw.Draw(overview_img)

    # 2. Generate and save tiles
    tile_count = 0
    print("Generating and saving tiles...")
    for i, (x1, y1, x2, y2) in enumerate(generate_tiles(img_w, img_h, tile_size, overlap)):
        tile_img = img.crop((x1, y1, x2, y2))
        tile_filename = f"tile_{i:04d}_{x1}_{y1}.png"
        tile_img.save(output_dir / tile_filename)
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=max(1, int(img_w / 1000)))
        
        tile_count += 1

    print(f"Successfully generated and saved {tile_count} tiles to {output_dir}")

    # 3. Create and save the side-by-side overview image
    original_img_rgb = img.convert("RGB")
    overview_img_rgb = overview_img.convert("RGB")
    
    side_by_side_img = Image.new('RGB', (img_w * 2, img_h))
    side_by_side_img.paste(original_img_rgb, (0, 0))
    side_by_side_img.paste(overview_img_rgb, (img_w, 0))

    overview_filename = "tiling_overview.png"
    overview_path = output_dir / overview_filename
    side_by_side_img.save(overview_path)
    print(f"Saved side-by-side overview image to {overview_path}")

    # 4. Create the guide markdown file
    create_visualization_guide(output_dir, overview_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the tiling process on a single image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output tiles and overview.")
    parser.add_argument("--tile_size", type=int, default=640, help="The size of each tile in pixels.")
    parser.add_argument("--overlap", type=float, default=0.2, help="The overlap ratio between adjacent tiles (0.0 to 1.0).")
    
    args = parser.parse_args()

    visualize_tiling(
        image_path=Path(args.image_path),
        output_dir=Path(args.output_dir),
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
