# /home/rune/dev/pid/validate_dataset.py
import os
import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


def validate_dataset(annotation_file: str, image_dir: str):
    """
    COCO 형식의 데이터셋을 검증하는 스크립트.

    수행하는 검증 작업:
    1. 이미지 파일 존재 여부 및 손상 여부 확인
    2. 어노테이션의 image_id, category_id 유효성 검사
    3. Bounding Box의 유효성 (w > 0, h > 0) 검사
    4. 어노테이션이 없는 이미지 목록 보고
    """
    print(f"🚀 데이터셋 검증을 시작합니다...")
    print(f"- 어노테이션 파일: {annotation_file}")
    print(f"- 이미지 디렉토리: {image_dir}\n")

    if not Path(annotation_file).exists():
        print(f"❌ 에러: 어노테이션 파일을 찾을 수 없습니다: {annotation_file}")
        return

    if not Path(image_dir).is_dir():
        print(f"❌ 에러: 이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        return

    coco = COCO(annotation_file)
    errors = defaultdict(list)
    
    # --- 1. 카테고리 및 이미지 ID 집합 생성 ---
    valid_category_ids = set(coco.getCatIds())
    valid_image_ids = set(coco.getImgIds())
    print(f"✅ 총 {len(valid_category_ids)}개의 카테고리, {len(valid_image_ids)}개의 이미지를 로드했습니다.")

    # --- 2. 이미지 파일 검증 ---
    print("\n[단계 1/3] 이미지 파일 검증 중...")
    images_with_annotations = set(coco.getAnnIds(imgIds=list(valid_image_ids)))
    annotated_image_ids = {coco.anns[ann_id]['image_id'] for ann_id in images_with_annotations}

    for img_id in tqdm(valid_image_ids, desc="이미지 검사"):
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        image_path = Path(image_dir) / file_name

        # 파일 존재 여부
        if not image_path.exists():
            errors["missing_images"].append(file_name)
            continue

        # 파일 손상 여부
        try:
            with Image.open(image_path) as img:
                img.verify()  # 이미지 헤더 및 구조 검사
        except Exception as e:
            errors["corrupt_images"].append(f"{file_name} (에러: {e})")

    # --- 3. 어노테이션 검증 ---
    print("\n[단계 2/3] 어노테이션 검증 중...")
    for ann_id in tqdm(coco.getAnnIds(), desc="어노테이션 검사"):
        ann = coco.loadAnns(ann_id)[0]

        # image_id 유효성
        if ann['image_id'] not in valid_image_ids:
            errors["invalid_ann_image_id"].append(f"어노테이션 ID {ann['id']} -> 존재하지 않는 이미지 ID {ann['image_id']}")

        # category_id 유효성
        if ann['category_id'] not in valid_category_ids:
            errors["invalid_ann_category_id"].append(f"어노테이션 ID {ann['id']} -> 존재하지 않는 카테고리 ID {ann['category_id']}")

        # Bbox 유효성
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                img_info = coco.loadImgs(ann['image_id'])[0]
                errors["invalid_bbox"].append(f"이미지 '{img_info['file_name']}'의 어노테이션 ID {ann['id']} (bbox: [w={w}, h={h}])")

    # --- 4. 어노테이션 없는 이미지 찾기 ---
    print("\n[단계 3/3] 어노테이션 없는 이미지 확인 중...")
    images_without_annotations = valid_image_ids - annotated_image_ids
    if images_without_annotations:
        for img_id in images_without_annotations:
            errors["images_without_annotations"].append(coco.loadImgs(img_id)[0]['file_name'])

    # --- 5. 결과 보고 ---
    print("\n🏁 데이터셋 검증 완료!\n")
    if not errors:
        print("🎉 축하합니다! 데이터셋에서 심각한 오류를 발견하지 못했습니다.")
        return

    print("🔥 다음 문제들이 발견되었습니다:\n")
    for error_type, error_list in errors.items():
        print(f"--- {error_type.replace('_', ' ').upper()} ({len(error_list)}개) ---")
        # 너무 많은 오류는 일부만 출력
        display_count = min(len(error_list), 20)
        for i in range(display_count):
            print(f"  - {error_list[i]}")
        if len(error_list) > display_count:
            print(f"  ... 외 {len(error_list) - display_count}개 더 있음")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO 데이터셋 무결성 검사기")
    parser.add_argument(
        "--ann",
        type=str,
        required=True,
        help="COCO 어노테이션 JSON 파일 경로",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="이미지 파일들이 있는 디렉토리 경로",
    )
    args = parser.parse_args()
    
    base_dir = Path(os.getcwd()).resolve()
    assets_dir = base_dir / "assets"
    
    ann_path = Path(args.ann)
    img_dir_path = Path(args.img_dir)
    img_dir_path = assets_dir / "image" / ""
    
    validate_dataset(args.ann, args.img_dir)
