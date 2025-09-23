import json
import random
from pathlib import Path
from collections import defaultdict, Counter
import shutil
from typing import Dict, List, Tuple, Union

from utils.file_utils import load_json_data, save_json_data


def stratified_split_coco(
    input_json_path: Union[str, Path],
    output_dir: Union[str, Path],
    image_dir: Union[str, Path, None] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    random_seed: int = 42,
    min_samples_per_category: int = 3,
    strategy: str = "dominant_category"  # "dominant_category" 또는 "multi_label"
) -> Dict[str, Dict[str, int]]:
    """
    카테고리별로 균등하게 분할하는 stratified split 함수
    
    Args:
        input_json_path: 입력 COCO JSON 파일
        output_dir: 출력 디렉토리
        image_dir: 이미지 디렉토리 (None이면 이미지 복사 안함)
        train_ratio: 훈련 데이터 비율
        val_ratio: 검증 데이터 비율
        random_seed: 랜덤 시드
        min_samples_per_category: 카테고리별 최소 샘플 수
        strategy: 분할 전략
            - "dominant_category": 가장 많은 어노테이션을 가진 카테고리 기준
            - "multi_label": 모든 카테고리를 고려한 멀티 라벨 방식
    
    Returns:
        각 split별 통계 정보
    """
    random.seed(random_seed)
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # 데이터 로드
    data = load_json_data
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Total categories: {len(categories)}")
    
    # 이미지별 어노테이션 매핑
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # 어노테이션이 있는 이미지만 필터링
    valid_images = [img for img in images if img['id'] in img_to_anns]
    print(f"Images with annotations: {len(valid_images)}")
    
    # 카테고리별 통계 출력
    _print_category_statistics(valid_images, img_to_anns, categories)
    
    # Stratified split 수행
    if strategy == "dominant_category":
        splits = _stratified_split_by_dominant_category(
            valid_images, img_to_anns, (train_ratio, val_ratio, test_ratio), 
            min_samples_per_category
        )
    else:  # multi_label
        splits = _stratified_split_multi_label(
            valid_images, img_to_anns, categories, (train_ratio, val_ratio, test_ratio),
            min_samples_per_category
        )
    
    # 결과 저장
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    (output_path / "annotations").mkdir(exist_ok=True)
    
    stats = {}
    for split_name, split_images in splits.items():
        # 해당 split의 어노테이션 수집
        split_annotations = []
        for img in split_images:
            split_annotations.extend(img_to_anns[img['id']])
        
        # ID 재할당
        split_images_copy = [img.copy() for img in split_images]
        split_annotations_copy = [ann.copy() for ann in split_annotations]
        
        _reassign_ids(split_images_copy, split_annotations_copy, img_to_anns)
        
        # JSON 생성
        split_data = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'images': split_images_copy,
            'annotations': split_annotations_copy,
            'categories': categories
        }
        
        # JSON 저장
        save_json_data(split_data, output_path / "annotations" / f'{split_name}.json')
        
        # 이미지 복사 (옵션)
        if image_dir:
            _copy_images_for_split(split_images, image_dir, output_path / split_name)
        
        # 통계 수집
        category_counts = Counter()
        for ann in split_annotations_copy:
            category_counts[ann['category_id']] += 1
        
        stats[split_name] = {
            'images': len(split_images_copy),
            'annotations': len(split_annotations_copy),
            'categories': dict(category_counts)
        }
        
        print(f"\n{split_name.upper()}:")
        print(f"  Images: {len(split_images_copy)}")
        print(f"  Annotations: {len(split_annotations_copy)}")
        print(f"  Category distribution: {dict(category_counts)}")
    
    # 분할 품질 검증
    _validate_stratified_split(stats, categories)
    
    return stats


def _stratified_split_by_dominant_category(
    images: List[Dict], 
    img_to_anns: Dict, 
    split_ratios: Tuple[float, float, float],
    min_samples_per_category: int
) -> Dict[str, List[Dict]]:
    """주요 카테고리 기준으로 stratified split"""
    
    # 각 이미지의 주요 카테고리 결정 (가장 많은 어노테이션을 가진 카테고리)
    category_to_images = defaultdict(list)
    
    for img in images:
        category_counts = Counter()
        for ann in img_to_anns[img['id']]:
            category_counts[ann['category_id']] += 1
        
        # 가장 많은 어노테이션을 가진 카테고리를 주요 카테고리로 설정
        dominant_category = category_counts.most_common(1)[0][0]
        category_to_images[dominant_category].append(img)
    
    splits = {"train": [], "val": [], "test": []}
    
    for category_id, cat_images in category_to_images.items():
        if len(cat_images) < min_samples_per_category:
            print(f"Warning: Category {category_id} has only {len(cat_images)} samples, assigning all to train")
            splits["train"].extend(cat_images)
            continue
        
        # 카테고리 내에서 랜덤 셔플
        random.shuffle(cat_images)
        
        # 비율에 따라 분할
        n_train = max(1, int(len(cat_images) * split_ratios[0]))
        n_val = max(1, int(len(cat_images) * split_ratios[1]))
        
        splits["train"].extend(cat_images[:n_train])
        splits["val"].extend(cat_images[n_train:n_train + n_val])
        splits["test"].extend(cat_images[n_train + n_val:])
        
        print(f"Category {category_id}: {len(cat_images)} total -> "
              f"train: {n_train}, val: {len(cat_images[n_train:n_train + n_val])}, "
              f"test: {len(cat_images[n_train + n_val:])}")
    
    return splits


def _stratified_split_multi_label(
    images: List[Dict], 
    img_to_anns: Dict, 
    categories: List[Dict],
    split_ratios: Tuple[float, float, float],
    min_samples_per_category: int
) -> Dict[str, List[Dict]]:
    """멀티 라벨을 고려한 stratified split (더 복잡하지만 정확함)"""
    
    # 각 이미지의 카테고리 집합 생성
    image_categories = {}
    for img in images:
        cats = set()
        for ann in img_to_anns[img['id']]:
            cats.add(ann['category_id'])
        image_categories[img['id']] = frozenset(cats)
    
    # 카테고리 조합별로 그룹화
    combination_to_images = defaultdict(list)
    for img in images:
        combination = image_categories[img['id']]
        combination_to_images[combination].append(img)
    
    splits = {"train": [], "val": [], "test": []}
    
    # 각 조합별로 분할
    for combination, comb_images in combination_to_images.items():
        if len(comb_images) < min_samples_per_category:
            splits["train"].extend(comb_images)
            continue
        
        random.shuffle(comb_images)
        
        n_train = max(1, int(len(comb_images) * split_ratios[0]))
        n_val = max(1, int(len(comb_images) * split_ratios[1]))
        
        splits["train"].extend(comb_images[:n_train])
        splits["val"].extend(comb_images[n_train:n_train + n_val])
        splits["test"].extend(comb_images[n_train + n_val:])
    
    return splits


def _print_category_statistics(images: List[Dict], img_to_anns: Dict, categories: List[Dict]):
    """카테고리별 통계 출력"""
    category_counts = Counter()
    for img in images:
        for ann in img_to_anns[img['id']]:
            category_counts[ann['category_id']] += 1
    
    print("\nCategory Statistics:")
    category_dict = {cat['id']: cat['name'] for cat in categories}
    for cat_id, count in sorted(category_counts.items()):
        cat_name = category_dict.get(cat_id, f"Unknown_{cat_id}")
        print(f"  Category {cat_id} ({cat_name}): {count} annotations")


def _reassign_ids(images: List[Dict], annotations: List[Dict], img_to_anns: Dict):
    """ID 재할당 (in-place)"""
    # 이미지 ID 매핑 생성
    old_to_new_image_id = {}
    for i, img in enumerate(images, 1):
        old_id = img['id']
        img['id'] = i
        old_to_new_image_id[old_id] = i
    
    # 어노테이션 ID 재할당 및 이미지 ID 업데이트
    for i, ann in enumerate(annotations, 1):
        ann['id'] = i
        ann['image_id'] = old_to_new_image_id[ann['image_id']]


def _copy_images_for_split(images: List[Dict], src_dir: Union[str, Path], dst_dir: Union[str, Path]):
    """Split별 이미지 복사"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(exist_ok=True)
    
    copied = 0
    for img in images:
        src_path = src_dir / img['file_name']
        dst_path = dst_dir / img['file_name']
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            print(f"Warning: Image not found: {src_path}")
    
    print(f"  Copied {copied}/{len(images)} images to {dst_dir}")


def _validate_stratified_split(stats: Dict, categories: List[Dict]):
    """분할 품질 검증"""
    print("\n" + "="*50)
    print("STRATIFIED SPLIT VALIDATION")
    print("="*50)
    
    category_dict = {cat['id']: cat['name'] for cat in categories}
    
    for cat_id in sorted(category_dict.keys()):
        cat_name = category_dict[cat_id]
        train_count = stats['train']['categories'].get(cat_id, 0)
        val_count = stats['val']['categories'].get(cat_id, 0)
        test_count = stats['test']['categories'].get(cat_id, 0)
        total = train_count + val_count + test_count
        
        if total > 0:
            train_pct = train_count / total * 100
            val_pct = val_count / total * 100
            test_pct = test_count / total * 100
            
            print(f"Category {cat_id} ({cat_name}):")
            print(f"  Train: {train_count:3d} ({train_pct:5.1f}%)")
            print(f"  Val:   {val_count:3d} ({val_pct:5.1f}%)")
            print(f"  Test:  {test_count:3d} ({test_pct:5.1f}%)")
            print(f"  Total: {total:3d}")
            print()


# 간단한 사용 함수
def split_dataset_stratified(
    merged_json: Union[str, Path],
    output_dir: Union[str, Path],
    image_dir: Union[str, Path] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2
):
    """
    가장 간단한 stratified split 사용법
    """
    return stratified_split_coco(
        input_json_path=merged_json,
        output_dir=output_dir,
        image_dir=image_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        strategy="dominant_category"  # 대부분의 경우에 적합
    )


if __name__ == "__main__":
    # 사용 예시
    stats = split_dataset_stratified(
        merged_json="assets/merged_VL_prepro.json",
        output_dir="assets/stratified_split",
        image_dir="assets/images",
        train_ratio=0.7,
        val_ratio=0.2
    )
    
    print("\nFinal Statistics:")
    for split_name, split_stats in stats.items():
        print(f"{split_name}: {split_stats['images']} images, {split_stats['annotations']} annotations")