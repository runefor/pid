import os
import json
from pathlib import Path


def save_json_data(data: dict, output_path: Path):
    """
    JSON 데이터를 지정된 경로에 저장합니다.

    Args:
        data (dict): 저장할 JSON 데이터.
        output_path (Path): 데이터를 저장할 파일 경로.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 데이터가 '{output_path}'에 저장되었습니다.")


def merge_coco_datasets(
    json_path1: Path,
    json_path2: Path,
    output_path: Path,
):
    """
    두 개의 COCO 형식 데이터셋을 병합합니다.
    - 이미지와 어노테이션 ID를 정수형으로 새로 부여하여 충돌을 방지하고 표준을 따릅니다.
    - 문자열 형식의 기존 image_id를 참조하여 어노테이션의 image_id를 올바르게 업데이트합니다.
    - 카테고리, 정보, 라이선스는 첫 번째 데이터셋을 기준으로 사용합니다.

    Args:
        json_path1 (Path): 첫 번째 COCO JSON 파일 경로.
        json_path2 (Path): 두 번째 COCO JSON 파일 경로.
        output_path (Path): 병합된 COCO JSON 파일을 저장할 경로.
    """
    print(f"🔄 병합 시작: '{json_path1.name}' + '{json_path2.name}'")

    # 1. 두 JSON 파일 로드
    with open(json_path1, "r", encoding="utf-8") as f:
        data1 = json.load(f)
    with open(json_path2, "r", encoding="utf-8") as f:
        data2 = json.load(f)

    # 2. 병합된 데이터를 담을 새로운 구조 생성
    merged_data = {
        "info": data1.get("info", {}),
        "licenses": data1.get("licenses", []),
        "categories": data1.get("categories", []),
        "images": [],
        "annotations": [],
    }

    # 3. 이미지 병합 및 ID 재할당 (정수형으로)
    # 기존의 문자열 ID와 새로 부여된 정수 ID를 매핑합니다.
    old_id_to_new_id = {}
    current_image_id = 1

    all_images = data1.get("images", []) + data2.get("images", [])
    for image in all_images:
        old_id = image["id"]
        if old_id not in old_id_to_new_id:
            old_id_to_new_id[old_id] = current_image_id
            image["id"] = current_image_id
            merged_data["images"].append(image)
            current_image_id += 1

    print(f"📊 이미지 병합 완료:")
    print(f"  - 총 {len(merged_data['images']):,}개의 고유 이미지")

    # 4. 어노테이션 병합 및 ID 재할당
    current_annotation_id = 1
    all_annotations = data1.get("annotations", []) + data2.get("annotations", [])
    for annotation in all_annotations:
        old_image_id = annotation["image_id"]
        # 매핑된 새로운 정수 image_id로 업데이트
        if old_image_id in old_id_to_new_id:
            annotation["image_id"] = old_id_to_new_id[old_image_id]
            annotation["id"] = current_annotation_id
            merged_data["annotations"].append(annotation)
            current_annotation_id += 1

    # 5. 병합된 JSON 파일 저장
    save_json_data(merged_data, output_path)

    print(f"\n✅ 병합 완료!")
    print(f"  - 총 이미지 수: {len(merged_data['images']):,}")
    print(f"  - 총 어노테이션 수: {len(merged_data['annotations']):,}")
    print(f"  - 저장된 파일: '{output_path}'")


def filter_coco_by_vendor(
    input_json_path: Path,
    output_json_path: Path,
    vendor_id: str,
):
    """
    COCO 데이터셋에서 특정 vendor의 이미지와 어노테이션만 필터링합니다.

    Args:
        input_json_path (Path): 원본 COCO JSON 파일 경로.
        output_json_path (Path): 필터링된 결과를 저장할 JSON 파일 경로.
        vendor_id (str): 필터링할 'vendor' 속성 값 (예: "V01").
    """
    print(f"\n🔄 '{vendor_id}' vendor 데이터 필터링 시작...")
    print(f"  - 입력 파일: '{input_json_path}'")

    # 1. 원본 데이터 로드
    with open(input_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # 2. vendor_id를 기준으로 '어노테이션' 필터링
    vendor_annotations = [
        ann for ann in coco_data.get("annotations", [])
        if ann.get("attributes", {}).get("vendor") == vendor_id
    ]

    if not vendor_annotations:
        print(f"⚠️ '{vendor_id}' vendor에 해당하는 어노테이션을 찾을 수 없습니다.")
        return

    print(f"📊 '{vendor_id}' vendor 어노테이션 {len(vendor_annotations):,}개 발견.")

    # 3. 필터링된 어노테이션에 해당하는 '이미지' ID 집합 생성
    vendor_image_ids = {ann["image_id"] for ann in vendor_annotations}
    print(f"   -> 해당 고유 이미지 {len(vendor_image_ids):,}개.")

    # 4. 해당 이미지 ID를 가진 이미지 정보 추출
    vendor_images = [
        img for img in coco_data.get("images", [])
        if img["id"] in vendor_image_ids
    ]

    # 5. 새로운 COCO 데이터 구조 생성
    filtered_data = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": vendor_images,
        "annotations": vendor_annotations,
    }

    # 6. 필터링된 JSON 파일 저장
    save_json_data(filtered_data, output_json_path)
    print(f"✅ '{vendor_id}' 데이터셋 저장 완료: '{output_json_path}'")

if __name__ == "__main__":
    base_dir = Path(os.getcwd()).resolve()
    assets_dir = base_dir / "assets"

    # merge_coco_datasets(
    #     json_path1=assets_dir / "merged_TL_prepro.json",
    #     json_path2=assets_dir / "merged_VL_prepro.json",
    #     output_path=assets_dir / "merged_dataset.json",
    # )

    # 'V01' vendor 데이터만 필터링하여 새로운 파일로 저장
    filter_coco_by_vendor(
        input_json_path=assets_dir / "merged_dataset.json",
        output_json_path=assets_dir / "merged_v01_dataset.json",
        vendor_id="V01",
    )
