from collections import defaultdict, Counter
import json
import random
from pathlib import Path
import os
import shutil
from typing import Dict, List, Tuple, Union, Optional
from enum import Enum
import numpy as np
import cv2
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class SplitStrategy(Enum):
    """분할 전략 열거형"""
    DOMINANT_CATEGORY = "dominant_category"
    MULTI_LABEL = "multi_label"
    HYBRID = "hybrid"
    ITERATIVE = "iterative"
    ITERATIVE_BY_ANNOTATION = "iterative_by_annotation"
    RANDOM = "random"
    
    def list():
        return list(map(lambda c: c.value, SplitStrategy))


class StratifiedDatasetSplitter:
    """
    COCO 형식 데이터셋을 위한 종합 Stratified Split 클래스
    
    세 가지 전략 지원:
    1. Dominant Category: 이미지당 가장 많은 어노테이션을 가진 카테고리 기준
    2. Multi-label: 모든 카테고리 조합을 고려한 정밀 분할
    3. Hybrid: Dominant + 소수 카테고리 보정
    4. Iterative: 전체 분포 균형을 최적화하는 반복적 분할 (가장 강력)
    5. Iterative by Annotation: 어노테이션 단위로 분할 (가장 정밀, 이미지 복제 발생)
    """
    
    def __init__(
        self,
        input_json_path: Union[str, Path],
        output_dir: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        random_seed: int = 42,
        strategy: Union[str, SplitStrategy] = SplitStrategy.HYBRID
    ):
        """
        Args:
            input_json_path: 입력 COCO JSON 파일 경로
            output_dir: 출력 디렉토리 경로
            image_dir: 이미지 디렉토리 경로 (이미지 복사용, 선택사항)
            train_ratio: 훈련 데이터 비율 (기본: 0.7)
            val_ratio: 검증 데이터 비율 (기본: 0.2)
            random_seed: 랜덤 시드
            strategy: 분할 전략 (dominant_category, multi_label, hybrid)
        """
        self.input_json_path = Path(input_json_path)
        self.output_dir = Path(output_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.random_seed = random_seed
        
        # 전략 설정
        if isinstance(strategy, str):
            self.strategy = SplitStrategy(strategy)
        else:
            self.strategy = strategy
        
        # 전략별 파라미터 (사용자가 수정 가능)
        self.strategy_params = {
            'rare_threshold': 50,           # Hybrid용: 소수 클래스 임계값
            'very_rare_threshold': 10,      # Hybrid용: 극소수 클래스 임계값  
            'min_samples_per_category': 3,  # 카테고리당 최소 샘플 수
            'min_samples_per_split': 2,     # Split당 최소 샘플 수
            'rebalance_tolerance': 0.1      # Hybrid용: 재조정 허용 오차
        }
        
        # 내부 데이터
        self.data = None
        self.images = None
        self.annotations = None
        self.categories = None
        self.img_to_anns = defaultdict(list)
        self.valid_images = None
        self.class_stats = {}
        
        random.seed(self.random_seed)
    
    def set_strategy_params(self, **params):
        """전략별 파라미터 설정"""
        self.strategy_params.update(params)
        return self
    
    def load_data(self):
        """데이터 로드 및 전처리"""
        print(f"📁 Loading data from: {self.input_json_path}")
        
        with open(self.input_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.images = self.data['images']
        self.annotations = self.data['annotations']
        self.categories = self.data['categories']
        
        print(f"📊 Dataset Overview:")
        print(f"   Total images: {len(self.images):,}")
        print(f"   Total annotations: {len(self.annotations):,}")
        print(f"   Total categories: {len(self.categories):,}")
        
        # 이미지별 어노테이션 매핑
        self.img_to_anns = defaultdict(list)
        for ann in self.annotations:
            self.img_to_anns[ann['image_id']].append(ann)
        
        # --- ID 일치 분석 코드 추가 ---
        all_image_ids = {img['id'] for img in self.images}
        ann_image_ids = set(self.img_to_anns.keys())
        matching_ids = all_image_ids.intersection(ann_image_ids)
        non_matching_ann_ids = ann_image_ids - all_image_ids

        print("\n🔍 Analyzing image and annotation ID matching:")
        print(f"   - Found {len(all_image_ids)} unique image IDs in 'images' list.")
        print(f"   - Found {len(ann_image_ids)} unique image IDs in 'annotations' list.")
        print(f"   - Found {len(matching_ids)} matching image IDs between them.")

        if len(ann_image_ids) > 0 and len(matching_ids) == 0:
            print("   - ⚠️ WARNING: No annotation 'image_id' matches any 'id' in the 'images' list.")
            if len(non_matching_ann_ids) > 0:
                print(f"   - Example non-matching annotation image_id(s): {list(non_matching_ann_ids)[:5]}")
        print()
        # --- ID 일치 분석 코드 끝 ---

        # 어노테이션이 있는 이미지만 필터링
        self.valid_images = [img for img in self.images if img['id'] in self.img_to_anns]
        print(f"   Valid images (with annotations): {len(self.valid_images):,}")
        
        # 클래스별 통계 분석
        self._analyze_class_statistics()
        
        return self
    
    def split(self) -> Dict[str, Dict[str, int]]:
        """선택된 전략으로 데이터 분할 실행"""
        if self.data is None:
            self.load_data()
        
        # --- 극소수 클래스 사전 분할 로직 ---
        pre_splits = {"train": [], "val": [], "test": []}
        pre_assigned_image_ids = set()

        # 이미지가 2개인 클래스 찾기
        for class_id, stats in self.class_stats.items():
            if stats['appears_in_images'] == 2:
                class_name = stats['name']
                print(f"   Applying pre-split for rare class '{class_name}' with 2 images.")
                
                # 해당 클래스를 포함하는 이미지 2개 찾기
                images_for_class = [
                    img for img in self.valid_images 
                    if any(ann['category_id'] == class_id for ann in self.img_to_anns[img['id']])
                ]
                
                # 이미 할당된 이미지는 건너뛰기
                images_to_assign = [img for img in images_for_class if img['id'] not in pre_assigned_image_ids]
                if len(images_to_assign) == 2:
                    random.shuffle(images_to_assign)
                    pre_splits["train"].append(images_to_assign[0])
                    pre_splits["val"].append(images_to_assign[1])
                    pre_assigned_image_ids.add(images_to_assign[0]['id'])
                    pre_assigned_image_ids.add(images_to_assign[1]['id'])

        # 사전 할당된 이미지를 제외한 유효 이미지 목록 업데이트
        original_valid_images = self.valid_images
        self.valid_images = [img for img in self.valid_images if img['id'] not in pre_assigned_image_ids]

        print(f"\n🔄 Applying {self.strategy.value} strategy...")
        
        # 전략별 분할 실행
        if self.strategy == SplitStrategy.DOMINANT_CATEGORY:
            splits = self._dominant_category_split()
        elif self.strategy == SplitStrategy.MULTI_LABEL:
            splits = self._multi_label_split()
        elif self.strategy == SplitStrategy.HYBRID:
            splits = self._hybrid_split()
        elif self.strategy == SplitStrategy.ITERATIVE:
            splits = self._iterative_split()
        elif self.strategy == SplitStrategy.RANDOM:
            splits = self._random_split()
        elif self.strategy == SplitStrategy.ITERATIVE_BY_ANNOTATION:
            return self._iterative_split_by_annotation()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # --- 사전 분할된 결과와 병합 ---
        for split_name in splits:
            splits[split_name].extend(pre_splits[split_name])

        # valid_images를 원상태로 복구
        self.valid_images = original_valid_images

        # 결과 저장 및 검증
        stats, _ = self._save_and_validate_splits(splits)
        
        return stats

    
    def _analyze_class_statistics(self):
        """클래스별 상세 통계 분석"""
        # 전체 클래스별 어노테이션 수
        total_class_counts = Counter()
        for img in self.valid_images:
            for ann in self.img_to_anns[img['id']]:
                total_class_counts[ann['category_id']] += 1
        
        # 클래스별 이미지 수 (해당 클래스가 dominant인 이미지)
        dominant_image_counts = Counter()
        for img in self.valid_images:
            class_counts = Counter()
            for ann in self.img_to_anns[img['id']]:
                class_counts[ann['category_id']] += 1
            
            if class_counts:
                dominant_class = class_counts.most_common(1)[0][0]
                dominant_image_counts[dominant_class] += 1
        
        # 통계 정리
        category_dict = {cat['id']: cat['name'] for cat in self.categories}
        
        for class_id in total_class_counts.keys():
            self.class_stats[class_id] = {
                'name': category_dict.get(class_id, f'Unknown_{class_id}'),
                'total_annotations': total_class_counts[class_id],
                'dominant_images': dominant_image_counts.get(class_id, 0),
                'appears_in_images': len([1 for img in self.valid_images 
                                        if any(ann['category_id'] == class_id 
                                              for ann in self.img_to_anns[img['id']])])
            }
        
        # 불균형 정도 계산
        counts = list(total_class_counts.values())
        if counts:
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"📈 Class Distribution Analysis:")
            print(f"   Most frequent class: {max_count:,} annotations")
            print(f"   Least frequent class: {min_count:,} annotations") 
            print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
            
            # 분포 추천
            if imbalance_ratio < 10:
                recommended = "dominant_category"
            elif imbalance_ratio < 100:
                recommended = "hybrid"
            else:
                recommended = "iterative_by_annotation" # 가장 강력한 iterative_by_annotation 추천
            
            if self.strategy.value != recommended:
                print(f"💡 Recommended strategy for this dataset: {recommended}")
                print(f"   (Currently using: {self.strategy.value})")
    
    def _dominant_category_split(self) -> Dict[str, List[Dict]]:
        """Dominant Category 기반 분할"""
        print("   Using dominant category per image...")
        
        # 클래스별 이미지 그룹화
        class_to_images = defaultdict(list)
        
        for img in self.valid_images:
            class_counts = Counter()
            for ann in self.img_to_anns[img['id']]:
                class_counts[ann['category_id']] += 1
            
            if class_counts:
                # 동점시 클래스 ID가 작은 것을 우선 선택 (일관성 위해)
                sorted_classes = sorted(class_counts.items(), 
                                      key=lambda x: (-x[1], x[0]))
                dominant_class = sorted_classes[0][0]
                class_to_images[dominant_class].append(img)
        
        return self._split_by_class_groups(class_to_images)
    
    def _multi_label_split(self) -> Dict[str, List[Dict]]:
        """Multi-label 기반 분할 (카테고리 조합 고려)"""
        print("   Using multi-label combinations...")
        
        # 이미지별 카테고리 조합 생성
        combination_to_images = defaultdict(list)
        
        for img in self.valid_images:
            categories_in_image = set()
            for ann in self.img_to_anns[img['id']]:
                categories_in_image.add(ann['category_id'])
            
            # frozenset으로 조합 생성 (순서 무관)
            combination = frozenset(categories_in_image)
            combination_to_images[combination].append(img)
        
        print(f"   Found {len(combination_to_images)} unique category combinations")
        
        # 조합별 분할
        splits = {"train": [], "val": [], "test": []}
        
        for combination, comb_images in combination_to_images.items():
            n_images = len(comb_images)
            
            if n_images < self.strategy_params['min_samples_per_category']:
                # 샘플이 너무 적으면 train에 모두 할당
                splits["train"].extend(comb_images)
                continue
            
            # 비율에 따라 분할
            random.shuffle(comb_images)
            
            n_train = max(1, int(n_images * self.train_ratio))
            n_val = max(1, int(n_images * self.val_ratio))
            
            splits["train"].extend(comb_images[:n_train])
            splits["val"].extend(comb_images[n_train:n_train + n_val])
            splits["test"].extend(comb_images[n_train + n_val:])
        
        return splits

    def _random_split(self) -> Dict[str, List[Dict]]:
        """단순 랜덤 분할"""
        print("   Using simple random split...")
        
        # 사전 할당된 이미지를 제외한 이미지 목록을 복사하여 사용
        images_to_split = self.valid_images.copy()
        random.shuffle(images_to_split)
        
        n_images = len(images_to_split)
        n_train = int(n_images * self.train_ratio)
        n_val = int(n_images * self.val_ratio)
        
        splits = {
            "train": images_to_split[:n_train],
            "val": images_to_split[n_train:n_train + n_val],
            "test": images_to_split[n_train + n_val:]
        }
        
        return splits
    
    def _hybrid_split(self) -> Dict[str, List[Dict]]:
        """Hybrid 방식: Dominant Category + 소수 클래스 보정"""
        print("   Using hybrid approach (dominant + rare class rebalancing)...")
        
        # 1단계: 클래스를 티어별로 분류
        rare_threshold = self.strategy_params['rare_threshold']
        very_rare_threshold = self.strategy_params['very_rare_threshold']
        
        class_tiers = self._classify_class_tiers(rare_threshold, very_rare_threshold)
        
        # 2단계: 이미지별 dominant class 결정 (소수 클래스 우선순위 부여)
        class_to_images = defaultdict(list)
        
        for img in self.valid_images:
            class_counts = Counter()
            for ann in self.img_to_anns[img['id']]:
                class_counts[ann['category_id']] += 1
            
            if class_counts:
                # 소수 클래스가 있으면 우선순위 부여
                rare_classes = [cls for cls in class_counts.keys() 
                              if cls in class_tiers['rare'] or cls in class_tiers['very_rare']]
                
                if rare_classes:
                    # 소수 클래스 중 가장 많은 것을 선택
                    dominant_class = max(rare_classes, key=lambda x: class_counts[x])
                else:
                    # 일반적인 dominant class 선택
                    dominant_class = class_counts.most_common(1)[0][0]
                
                class_to_images[dominant_class].append(img)
        
        # 3단계: 티어별로 다른 분할 전략 적용
        return self._split_by_tiers(class_to_images, class_tiers)

    def _iterative_split(self) -> Dict[str, List[Dict]]:
        """
        반복적 계층화 분할 (Iterative Stratification)
        전체 카테고리 분포를 최적화하여 이미지를 하나씩 할당합니다.
        """
        print("Using iterative stratification for optimal balance...")

        # 1. 초기 설정
        unassigned_images = self.valid_images.copy() # 사전 할당된 이미지가 제외된 리스트
        random.shuffle(unassigned_images)
        
        # 이미지 ID -> 이미지 정보, 카테고리 Set 매핑
        img_id_to_info = {img['id']: img for img in unassigned_images}
        img_id_to_cats = {
            img_id: frozenset(ann['category_id'] for ann in anns)
            for img_id, anns in self.img_to_anns.items()
        }

        # 2. 목표 분포 계산 (어노테이션 수 기준)
        target_dist = {
            'train': self.train_ratio,
            'val': self.val_ratio,
            'test': self.test_ratio
        }
        target_ann_counts = defaultdict(lambda: defaultdict(float))
        for cat_id, stats in self.class_stats.items():
            total_anns = stats['total_annotations']
            for split in target_dist:
                target_ann_counts[cat_id][split] = total_anns * target_dist[split]

        # 3. 현재 분할 상태 초기화
        splits = {"train": [], "val": [], "test": []}
        current_ann_counts = defaultdict(lambda: Counter())
        
        # 4. 이미지 반복 할당
        pbar = range(len(unassigned_images))
        for _ in pbar:
            # 가장 희귀한 카테고리를 가진 이미지를 찾음
            # (전체 데이터셋에서 가장 적게 나타나는 카테고리)
            min_cat_count = float('inf')
            best_img_id = -1
            
            # 아직 할당되지 않은 이미지 중에서 선택
            unassigned_ids = list(img_id_to_info.keys())
            if not unassigned_ids: break

            for img_id in unassigned_ids:
                cats_in_img = img_id_to_cats.get(img_id, set())
                if not cats_in_img: continue
                
                # 이미지에 포함된 카테고리 중 가장 희귀한 카테고리의 등장 횟수
                rarest_cat_count_in_img = min(self.class_stats[cat_id]['appears_in_images'] for cat_id in cats_in_img)
                
                if rarest_cat_count_in_img < min_cat_count:
                    min_cat_count = rarest_cat_count_in_img
                    best_img_id = img_id
            
            if best_img_id == -1: # 남은 이미지가 어노테이션이 없는 경우
                best_img_id = unassigned_ids[0]

            # 5. 최적의 split 찾기
            # 이미지를 각 split에 추가했을 때 목표 분포와의 차이가 가장 적은 곳을 선택
            best_split = ''
            min_diff = float('inf')
            
            img_cats = img_id_to_cats.get(best_img_id, set())
            
            for split_name in splits.keys():
                diff = 0
                # 이 split에 이미지를 추가했을 때, 각 카테고리의 목표 달성률을 계산
                for cat_id in img_cats:
                    # 이 split에 있는 해당 카테고리의 현재 어노테이션 수
                    current_ann_count = current_ann_counts[split_name].get(cat_id, 0)
                    total_ann_for_cat = self.class_stats[cat_id]['total_annotations']
                    
                    # 이 split의 목표 어노테이션 수
                    target_ratio = target_dist[split_name]
                    target_ann_count_for_split = total_ann_for_cat * target_ratio
                    
                    # 목표 대비 현재 얼마나 채워졌는지 비율을 계산
                    # 이 값이 작을수록 해당 split/category에 이미지가 더 필요하다는 의미
                    # +1을 하여 0으로 나누는 것을 방지하고, 아직 할당되지 않은 경우를 처리
                    fulfillment_ratio = (current_ann_count + 1) / (target_ann_count_for_split + 1)
                    diff += fulfillment_ratio
                
                if diff < min_diff:
                    min_diff = diff
                    best_split = split_name

            # 6. 이미지 할당 및 상태 업데이트
            splits[best_split].append(img_id_to_info[best_img_id])
            for cat_id in img_cats:
                current_ann_counts[best_split][cat_id] += 1
            del img_id_to_info[best_img_id]

        return splits
    
    def _iterative_split_by_annotation(self) -> Dict[str, Dict[str, int]]:
        """
        어노테이션 단위의 반복적 계층화 분할.
        이미지를 복제하여 각 split에 필요한 어노테이션만 포함시킵니다.
        """
        print("Using iterative stratification by annotation (most precise)...")

        # 1. 목표 비율 설정
        target_dist = {
            'train': self.train_ratio,
            'val': self.val_ratio,
            'test': self.test_ratio
        }

        # 2. 각 카테고리별로 어노테이션 분할
        split_annotations = defaultdict(list)
        cat_to_anns = defaultdict(list)
        for ann in self.annotations:
            cat_to_anns[ann['category_id']].append(ann)

        for cat_id, anns in cat_to_anns.items():
            random.shuffle(anns)
            n_anns = len(anns)
            n_train = int(n_anns * self.train_ratio)
            n_val = int(n_anns * self.val_ratio)
            
            split_annotations['train'].extend(anns[:n_train])
            split_annotations['val'].extend(anns[n_train:n_train + n_val])
            split_annotations['test'].extend(anns[n_train + n_val:])

        # 3. 분할된 어노테이션을 기반으로 최종 데이터 구조 생성 및 저장
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)
        
        stats = {}
        img_id_map = {img['id']: img for img in self.images}

        for split_name, anns in split_annotations.items():
            if not anns:
                continue

            # 이 split에 포함된 이미지 ID 수집
            image_ids_in_split = {ann['image_id'] for ann in anns}
            split_images = [img_id_map[img_id] for img_id in image_ids_in_split if img_id in img_id_map]

            # ID 재할당
            split_images_copy = [img.copy() for img in split_images]
            split_annotations_copy = [ann.copy() for ann in anns]
            self._reassign_ids(split_images_copy, split_annotations_copy)

            # JSON 저장
            split_data = {
                'info': self.data.get('info', {}),
                'licenses': self.data.get('licenses', []),
                'images': split_images_copy,
                'annotations': split_annotations_copy,
                'categories': self.categories
            }
            json_path = self.output_dir / "annotations" / f'{split_name}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)

            # 이미지 복사 (옵션)
            if self.image_dir:
                self._copy_images(split_images, self.output_dir / split_name)

            # 통계 수집
            stats[split_name] = {
                'images': len(split_images_copy),
                'annotations': len(split_annotations_copy),
                'categories': dict(Counter(ann['category_id'] for ann in split_annotations_copy))
            }
            print(f"   {split_name}: {len(split_images_copy):,} images, {len(split_annotations_copy):,} annotations")

        # 검증 리포트 생성
        self._generate_validation_report(stats, self.output_dir / 'validation_report.md', {k: {ann['image_id'] for ann in v} for k, v in split_annotations.items()})
        # 통계와 이미지 ID 셋을 함께 반환
        return stats, {k: {ann['image_id'] for ann in v} for k, v in split_annotations.items()}

    def _classify_class_tiers(self, rare_threshold, very_rare_threshold):
        """클래스를 티어별로 분류"""
        tiers = {
            'major': [],
            'medium': [],
            'rare': [],
            'very_rare': []
        }
        
        for class_id, stats in self.class_stats.items():
            count = stats['total_annotations']
            
            if count < very_rare_threshold:
                tiers['very_rare'].append(class_id)
            elif count < rare_threshold:
                tiers['rare'].append(class_id)
            elif count < 1000:
                tiers['medium'].append(class_id)
            else:
                tiers['major'].append(class_id)
        
        print(f"   Class tiers: Major({len(tiers['major'])}), "
              f"Medium({len(tiers['medium'])}), "
              f"Rare({len(tiers['rare'])}), "
              f"VeryRare({len(tiers['very_rare'])})")
        
        return tiers
    
    def _split_by_class_groups(self, class_to_images):
        """클래스별 이미지 그룹을 분할"""
        splits = {"train": [], "val": [], "test": []}
        
        for class_id, images in class_to_images.items():
            if len(images) < self.strategy_params['min_samples_per_category']:
                splits["train"].extend(images)
                continue
            
            random.shuffle(images)
            
            n_train = max(1, int(len(images) * self.train_ratio))
            n_val = max(1, int(len(images) * self.val_ratio))
            
            splits["train"].extend(images[:n_train])
            splits["val"].extend(images[n_train:n_train + n_val])
            splits["test"].extend(images[n_train + n_val:])
        
        return splits
    
    def _split_by_tiers(self, class_to_images, class_tiers):
        """티어별로 다른 전략으로 분할"""
        splits = {"train": [], "val": [], "test": []}
        
        for tier_name, class_ids in class_tiers.items():
            for class_id in class_ids:
                images = class_to_images.get(class_id, [])
                if not images:
                    continue
                
                n_images = len(images)
                random.shuffle(images)
                
                if tier_name == 'very_rare':
                    if n_images >= self.strategy_params['min_samples_per_category']: # 기본값 3 이상
                        # 3개일 경우: train 1, val 1, test 1
                        # 4개일 경우: train 2, val 1, test 1
                        # 5개일 경우: train 3, val 1, test 1
                        splits["test"].extend(images[:1])
                        splits["val"].extend(images[1:2])
                        splits["train"].extend(images[2:])
                    else:
                        # 샘플 수가 부족하면 모두 train에 할당
                        splits["train"].extend(images)
                        
                elif tier_name == 'rare':
                    # 소수: 최소 샘플 수를 만족하면 보수적 분할, 아니면 모두 train
                    if n_images >= self.strategy_params['min_samples_per_category']: # 기본값 3 이상
                        # 보수적 분할 (예: 80:10:10) 시도, 각 split에 최소 1개 보장
                        n_val = max(1, int(n_images * 0.1))
                        n_test = max(1, int(n_images * 0.1))
                        n_train = n_images - n_val - n_test
                        
                        splits["test"].extend(images[:n_test])
                        splits["val"].extend(images[n_test:n_test + n_val])
                        splits["train"].extend(images[n_test + n_val:])
                    else:
                        splits["train"].extend(images)
                        
                else:
                    # 일반: 표준 비율 분할
                    n_train = int(n_images * self.train_ratio)
                    n_val = int(n_images * self.val_ratio)
                    
                    splits["train"].extend(images[:n_train])
                    splits["val"].extend(images[n_train:n_train + n_val])
                    splits["test"].extend(images[n_train + n_val:])
        
        return splits
    
    def _save_and_validate_splits(self, splits):
        """분할 결과 저장 및 검증"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)
        
        stats = {}
        all_split_image_ids = defaultdict(set)
        
        for split_name, split_images in splits.items():
            if not split_images:
                continue
            
            # 해당 split의 어노테이션 수집
            split_annotations = []
            for img in split_images:
                split_annotations.extend(self.img_to_anns[img['id']])
                all_split_image_ids[split_name].add(img['id'])
            
            # ID 재할당
            split_images_copy = [img.copy() for img in split_images]
            split_annotations_copy = [ann.copy() for ann in split_annotations]
            
            self._reassign_ids(split_images_copy, split_annotations_copy)
            
            # JSON 저장
            split_data = {
                'info': self.data.get('info', {}),
                'licenses': self.data.get('licenses', []),
                'images': split_images_copy,
                'annotations': split_annotations_copy,
                'categories': self.categories
            }
            
            json_path = self.output_dir / "annotations" / f'{split_name}.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            # 이미지 복사 (옵션)
            if self.image_dir:
                self._copy_images(split_images, self.output_dir / split_name)
            
            # 통계 수집
            class_counts = Counter()
            for ann in split_annotations_copy:
                class_counts[ann['category_id']] += 1
            
            stats[split_name] = {
                'images': len(split_images_copy),
                'annotations': len(split_annotations_copy),
                'categories': dict(class_counts)
            }
            
            print(f"   {split_name}: {len(split_images_copy):,} images, "
                  f"{len(split_annotations_copy):,} annotations")
        
        # 검증 리포트 생성
        report_path = self.output_dir / 'validation_report.md'
        self._generate_validation_report(stats, report_path, all_split_image_ids)
        
        print(f"\n✅ Split completed! Results saved to: {self.output_dir}")
        print(f"   📊 Validation report saved to: {report_path}")
        return stats, all_split_image_ids
    
    def _reassign_ids(self, images, annotations):
        """ID 재할당"""
        old_to_new_image_id = {}
        for i, img in enumerate(images, 1):
            old_id = img['id']
            img['id'] = i
            old_to_new_image_id[old_id] = i
        
        for i, ann in enumerate(annotations, 1):
            ann['id'] = i
            ann['image_id'] = old_to_new_image_id[ann['image_id']]
    
    def _copy_images(self, images, dst_dir):
        """이미지 복사"""
        if not self.image_dir or not self.image_dir.exists():
            print(f"   Warning: Image directory not found: {self.image_dir}")
            return
        
        dst_dir.mkdir(exist_ok=True)
        
        copied = 0
        for img in images:
            src_path = self.image_dir / img['file_name']
            dst_path = dst_dir / img['file_name']
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1
        
        print(f"   Copied {copied}/{len(images)} images")
    
    def _generate_validation_report(self, stats: Dict[str, Dict], report_path: Path, all_split_image_ids: Dict[str, set]):
        """검증 리포트를 마크다운 파일로 생성"""
        report_lines = []

        report_lines.append(f"# 📋 Stratified Split Validation Report ({self.strategy.value.upper()})")
        report_lines.append("\n")

        category_dict = {cat['id']: cat['name'] for cat in self.categories}

        # 전체 통계
        total_images = sum(split_stats['images'] for split_stats in stats.values())
        unique_image_ids = set.union(*all_split_image_ids.values()) if all_split_image_ids else set()
        unique_images_count = len(unique_image_ids)
        total_annotations = sum(split_stats['annotations'] for split_stats in stats.values())
        duplication_rate = (total_images / unique_images_count - 1) * 100 if unique_images_count > 0 else 0


        if total_images == 0:
            report_lines.append("No data to report.")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(report_lines))
            return

        # --- 분할 비율 정보 추가 ---
        train_imgs = stats.get('train', {}).get('images', 0)
        val_imgs = stats.get('val', {}).get('images', 0)
        test_imgs = stats.get('test', {}).get('images', 0)
        
        # 분할된 총 이미지 수 기준 실제 비율 계산 (중복 포함)
        actual_train_ratio = train_imgs / total_images * 100 if total_images > 0 else 0
        actual_val_ratio = val_imgs / total_images * 100 if total_images > 0 else 0
        actual_test_ratio = test_imgs / total_images * 100 if total_images > 0 else 0

        report_lines.append("## 🎯 Split Ratio Summary\n")
        report_lines.append(f"- **Target Ratio (Train:Val:Test)**: {self.train_ratio * 100:.0f} : {self.val_ratio * 100:.0f} : {self.test_ratio * 100:.0f}")
        report_lines.append(f"- **Actual Ratio (based on image counts per split)**: {actual_train_ratio:.1f} : {actual_val_ratio:.1f} : {actual_test_ratio:.1f}\n")
        # --- 분할 비율 정보 추가 끝 ---

        report_lines.append("## Overall Distribution\n")
        report_lines.append(f"- **Total Categories**: {len(self.categories):,}\n")
        report_lines.append(f"- **Unique Images**: {unique_images_count:,}")
        if duplication_rate > 0.1:
            report_lines.append(f"- **Image Duplication Rate**: {duplication_rate:.1f}% (An image can appear in multiple splits)\n")
        
        report_lines.append("| Split | Images | Image % | Annotations | Annotation % |")
        report_lines.append("|:------|-------:|--------:|------------:|-------------:|")
        for split_name, split_stats in stats.items():
            # 이미지 %는 분할된 총 이미지 수 대비로 계산
            img_pct = split_stats['images'] / total_images * 100 if total_images > 0 else 0
            ann_pct = split_stats['annotations'] / total_annotations * 100 if total_annotations > 0 else 0
            report_lines.append(f"| {split_name} | {split_stats['images']:,} | {img_pct:.1f}% | {split_stats['annotations']:,} | {ann_pct:.1f}% |")
        report_lines.append("\n")

        # 카테고리별 분포 (상위 20개만)
        report_lines.append("## Category Distribution (All Categories by Total Annotations)\n")
        report_lines.append("| Category | Train (Count) | Train (%) | Val (Count) | Val (%) | Test (Count) | Test (%) | Total |")
        report_lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|")

        # 전체 카테고리를 어노테이션 수 기준으로 정렬
        all_categories = set()
        for split_stats in stats.values():
            all_categories.update(split_stats['categories'].keys())

        category_totals = defaultdict(int)
        for cat_id in all_categories: # 모든 카테고리의 전체 어노테이션 수 계산
            total = sum(stats.get(split, {}).get('categories', {}).get(cat_id, 0)
                        for split in ['train', 'val', 'test'])
            category_totals[cat_id] = total

        # 상위 20개 카테고리 출력
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)

        for cat_id, total in sorted_categories:
            cat_name = category_dict.get(cat_id, f'Unknown_{cat_id}')
            train_count = stats.get('train', {}).get('categories', {}).get(cat_id, 0)
            val_count = stats.get('val', {}).get('categories', {}).get(cat_id, 0)
            test_count = stats.get('test', {}).get('categories', {}).get(cat_id, 0)

            train_pct = (train_count / total * 100) if total > 0 else 0
            val_pct = (val_count / total * 100) if total > 0 else 0
            test_pct = (test_count / total * 100) if total > 0 else 0

            report_lines.append(f"| {cat_name} | {train_count:,} | {train_pct:.1f}% | {val_count:,} | {val_pct:.1f}% | {test_count:,} | {test_pct:.1f}% | {total:,} |")

        # 분포 품질 분석
        self._add_distribution_quality_analysis(report_lines, stats, category_totals, category_dict)

        # 문제 카테고리 분석 (Val/Test에 샘플이 거의 없는 경우)
        problem_categories = []
        # min_samples_threshold를 2로 설정하여 0, 1, 2개인 경우를 찾습니다.
        min_samples_threshold = 2
        for cat_id in all_categories:
            val_count = stats.get('val', {}).get('categories', {}).get(cat_id, 0)
            test_count = stats.get('test', {}).get('categories', {}).get(cat_id, 0)

            if val_count <= min_samples_threshold or test_count <= min_samples_threshold:
                train_count = stats.get('train', {}).get('categories', {}).get(cat_id, 0)
                total = category_totals[cat_id]
                problem_categories.append({
                    'id': cat_id,
                    'name': category_dict.get(cat_id, f'Unknown_{cat_id}'),
                    'train': train_count,
                    'val': val_count,
                    'test': test_count,
                    'total': total
                })
        
        if problem_categories:
            report_lines.append("\n## ⚠️ Problem Category Analysis (Low Samples in Val/Test)\n")
            report_lines.append(f"Categories with **{min_samples_threshold} or fewer** samples in validation or test splits.\n")
            report_lines.append("| Category | Train | Val | Test | Total |")
            report_lines.append("|:---|---:|---:|---:|---:|")
            # Val 샘플 수, Test 샘플 수, 전체 샘플 수 순으로 정렬
            sorted_problem_cats = sorted(problem_categories, key=lambda x: (x['val'], x['test'], x['total']))
            for cat in sorted_problem_cats:
                report_lines.append(f"| {cat['name']} | {cat['train']:,} | **{cat['val']:,}** | **{cat['test']:,}** | {cat['total']:,} |")

        # 파일 저장
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

    def _add_distribution_quality_analysis(self, report_lines: list, stats: dict, category_totals: dict, category_dict: dict):
        """분포 품질 분석 섹션을 리포트에 추가"""
        target_ratios = {
            'train': self.train_ratio,
            'val': self.val_ratio,
            'test': self.test_ratio
        }

        category_divergence = {}
        for cat_id, total in category_totals.items():
            if total == 0:
                continue
            
            divergence = 0
            for split_name, target_ratio in target_ratios.items():
                actual_count = stats.get(split_name, {}).get('categories', {}).get(cat_id, 0)
                actual_ratio = actual_count / total
                # 목표 비율과의 절대적인 차이를 합산
                divergence += abs(actual_ratio - target_ratio)
            
            category_divergence[cat_id] = divergence

        # 편차 점수가 높은 순으로 정렬
        sorted_divergence = sorted(category_divergence.items(), key=lambda x: x[1], reverse=True)

        report_lines.append("\n## ⚠️ Distribution Quality Analysis (Top 10 Most Skewed Categories)\n")
        report_lines.append("This section highlights categories whose distribution significantly deviates from the target ratio (e.g., 70:20:10).")
        report_lines.append("**Divergence Score**: A measure of how far the actual distribution is from the target. Higher is worse. (Max: 2.0)\n")
        report_lines.append("| Category | Train % | Val % | Test % | Divergence Score |")
        report_lines.append("|:---|---:|---:|---:|---:|")

        for cat_id, divergence_score in sorted_divergence[:10]:
            total = category_totals[cat_id]
            cat_name = category_dict.get(cat_id, f'Unknown_{cat_id}')
            train_pct = (stats.get('train', {}).get('categories', {}).get(cat_id, 0) / total * 100) if total > 0 else 0
            val_pct = (stats.get('val', {}).get('categories', {}).get(cat_id, 0) / total * 100) if total > 0 else 0
            test_pct = (stats.get('test', {}).get('categories', {}).get(cat_id, 0) / total * 100) if total > 0 else 0

            report_lines.append(f"| {cat_name} | {train_pct:.1f}% | {val_pct:.1f}% | {test_pct:.1f}% | **{divergence_score:.3f}** |")


def analyze_rare_class_locality(
    splitter: StratifiedDatasetSplitter,
    output_dir: Union[str, Path],
    source_dirs: List[Union[str, Path]],
    num_examples_per_class: int = 3
):
    """
    소수 카테고리의 지역성(locality)을 분석하고 시각화합니다.
    각 소수/극소수 카테고리에 대해, 해당 어노테이션이 포함된 이미지를 찾아
    그 분포를 시각화한 이미지 파일을 저장합니다.

    Args:
        splitter: 데이터가 로드된 StratifiedDatasetSplitter 인스턴스.
        output_dir: 시각화 결과물을 저장할 디렉토리.
        source_dirs: 원본 이미지 파일이 있는 디렉토리 목록.
        num_examples_per_class: 클래스당 시각화할 최대 이미지 예시 수.
    """
    if splitter.data is None:
        splitter.load_data()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🔬 Analyzing locality of rare classes. Visualizations will be saved to: {output_dir}")

    # 1. 소수/극소수 카테고리 식별
    rare_threshold = splitter.strategy_params['rare_threshold']
    very_rare_threshold = splitter.strategy_params['very_rare_threshold']
    class_tiers = splitter._classify_class_tiers(rare_threshold, very_rare_threshold)
    rare_class_ids = class_tiers['rare'] + class_tiers['very_rare']

    if not rare_class_ids:
        print("   No rare classes found to analyze.")
        return

    category_dict = {cat['id']: cat['name'] for cat in splitter.categories}
    image_map = {img['id']: img for img in splitter.images}

    # 원본 이미지 경로를 빠르게 찾기 위한 매핑 생성
    print("   Scanning source image directories...")
    image_path_map = {}
    for s_dir in source_dirs:
        s_dir = Path(s_dir)
        if not s_dir.exists(): continue
        for img_path in s_dir.rglob('*.png'):
            image_path_map[img_path.name] = img_path
    print(f"   Found {len(image_path_map)} unique source images.")

    # 2. 각 소수 카테고리별로 이미지 내 분포 시각화
    for class_id in rare_class_ids:
        class_name_safe = category_dict.get(class_id, f"Unknown_{class_id}").replace("/", "_").replace("@", "_")
        print(f"   Analyzing class: {class_name_safe} (ID: {class_id})")

        # 해당 클래스를 포함하는 이미지와 어노테이션 정보 수집
        images_with_class = []
        for img_id, anns in splitter.img_to_anns.items():
            class_anns = [ann for ann in anns if ann['category_id'] == class_id]
            if class_anns:
                images_with_class.append({
                    'image_info': image_map[img_id],
                    'annotations': class_anns,
                    'count': len(class_anns)
                })

        if not images_with_class:
            continue

        # 어노테이션 개수가 많은 순으로 정렬
        images_with_class.sort(key=lambda x: x['count'], reverse=True)

        # 상위 예시 이미지 시각화
        for i, item in enumerate(images_with_class[:num_examples_per_class]):
            img_info = item['image_info']
            # 이미지 파일명에서 경로를 제거하고 안전한 파일명으로 만듭니다.
            base_filename = Path(img_info['file_name']).name
            save_path = output_dir / f"{class_name_safe}_example_{i+1}_{base_filename}.png"
            
            # 원본 이미지 경로를 함께 전달
            source_image_path = image_path_map.get(base_filename)
            _visualize_annotations_on_image(img_info, item['annotations'], class_name_safe, save_path, source_image_path)

# 편의 함수들
def create_splitter(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    strategy: str = "hybrid",
    **kwargs
) -> StratifiedDatasetSplitter:
    """간편한 splitter 생성 함수"""
    return StratifiedDatasetSplitter(
        input_json_path=input_json,
        output_dir=output_dir,
        strategy=strategy,
        **kwargs
    )


def quick_split(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    strategy: str = "hybrid",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    **kwargs
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, set]]:
    """원샷 분할 함수"""
    splitter = create_splitter(
        input_json=input_json,
        output_dir=output_dir,
        strategy=strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        **kwargs
    )
    
    # splitter.split()은 이제 (stats, image_ids) 튜플을 반환할 수 있음
    result = splitter.split()
    if isinstance(result, tuple) and len(result) == 2:
        stats, image_ids = result
    else: # 이전 버전 호환성
        stats = result
        image_ids = {} # ID 정보가 없는 경우 빈 딕셔너리

    return stats, image_ids



# 추가 유틸리티 함수들
def compare_strategies(
    input_json: Union[str, Path],
    output_base_dir: Union[str, Path] = "strategy_comparison",
    strategies: List[str] = None
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """여러 전략을 비교하는 함수"""
    if strategies is None:
        strategies = SplitStrategy.list()
    
    results = {}
    all_image_ids_by_strategy = {}
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    for strategy in strategies:
        print(f"\n🔄 Comparing strategy: {strategy}")
        try:
            stats, image_ids = quick_split(
                input_json=input_json,
                output_dir=output_base / strategy,
                strategy=strategy
            )
            results[strategy] = stats
            all_image_ids_by_strategy[strategy] = image_ids
            
        except Exception as e:
            print(f"❌ Error with {strategy}: {e}")
            results[strategy] = None
            all_image_ids_by_strategy[strategy] = None
    
    # 비교 리포트 생성
    report_path = output_base / "comparison_report.md"
    _generate_comparison_report(results, report_path, all_image_ids_by_strategy)
    print(f"\n📊 Strategy comparison report saved to: {report_path}")
    return results


def _generate_comparison_report(results: Dict, output_path: Path, all_image_ids: Dict[str, Dict[str, set]]):
    """전략 비교 리포트를 생성하고 마크다운 파일로 저장"""
    report_lines = []
    console_lines = []

    # --- 콘솔용 헤더 ---
    console_lines.append(f"\n" + "="*90)
    console_lines.append("📊 STRATEGY COMPARISON REPORT")
    console_lines.append("="*90)
    console_lines.append(f"{'Strategy':<25} {'Train Imgs':<12} {'Val Imgs':<12} {'Test Imgs':<12} {'Unique Imgs':<13} {'Duplication':<12}")
    console_lines.append("-" * 90)

    # --- 마크다운용 헤더 ---
    report_lines.append("# 📊 Strategy Comparison Report")
    report_lines.append("")
    report_lines.append("| Strategy | Train Images | Val Images | Test Images | Unique Images | Duplication |")
    report_lines.append("|:---|---:|---:|---:|---:|---:|")
    
    for strategy, stats in results.items():
        if stats is None:
            console_lines.append(f"{strategy:<25} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12} {'ERROR':<15}")
            report_lines.append(f"| {strategy} | ERROR | ERROR | ERROR | ERROR |")
            continue
            
        train_imgs = stats.get('train', {}).get('images', 0)
        val_imgs = stats.get('val', {}).get('images', 0) 
        test_imgs = stats.get('test', {}).get('images', 0)
        
        image_ids = all_image_ids.get(strategy, {})
        unique_image_ids = set.union(*image_ids.values()) if image_ids else set()
        unique_imgs_count = len(unique_image_ids)
        total_imgs = train_imgs + val_imgs + test_imgs
        duplication_rate = (total_imgs / unique_imgs_count - 1) * 100 if unique_imgs_count > 0 else 0
        duplication_str = f"{duplication_rate:.1f}%"

        console_lines.append(f"{strategy:<25} {train_imgs:<12,} {val_imgs:<12,} {test_imgs:<12,} {unique_imgs_count:<13,} {duplication_str:<12}")
        report_lines.append(f"| {strategy} | {train_imgs:,} | {val_imgs:,} | {test_imgs:,} | {unique_imgs_count:,} | {duplication_str} |")

    # 콘솔에 출력
    print("\n".join(console_lines))

    # 파일에 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))


def analyze_dataset_characteristics(input_json: Union[str, Path]) -> Dict:
    """데이터셋 특성 분석하여 최적 전략 추천"""
    print("🔍 Analyzing dataset characteristics...")
    
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # 기본 통계
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    valid_images = [img for img in images if img['id'] in img_to_anns]
    
    # 클래스별 분포
    class_counts = Counter()
    for img in valid_images:
        for ann in img_to_anns[img['id']]:
            class_counts[ann['category_id']] += 1
    
    # 이미지당 평균 카테고리 수
    avg_categories_per_image = sum(len(anns) for anns in img_to_anns.values()) / len(valid_images)
    
    # 불균형 정도
    counts = list(class_counts.values())
    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # 카테고리 조합 복잡도
    combinations = set()
    for img in valid_images:
        categories_in_img = frozenset(ann['category_id'] for ann in img_to_anns[img['id']])
        combinations.add(categories_in_img)
    
    characteristics = {
        'total_images': len(valid_images),
        'total_categories': len(categories),
        'imbalance_ratio': imbalance_ratio,
        'avg_categories_per_image': avg_categories_per_image,
        'unique_combinations': len(combinations),
        'class_distribution': dict(class_counts)
    }
    
    # 전략 추천
    if imbalance_ratio < 10:
        recommended_strategy = "dominant_category"
        reason = "Low imbalance, simple approach sufficient"
    elif imbalance_ratio < 100:
        recommended_strategy = "hybrid" 
        reason = "Moderate imbalance, hybrid approach optimal"
    else:
        recommended_strategy = "iterative_by_annotation"
        reason = "High imbalance or complex co-occurrence, iterative_by_annotation approach recommended for best balance"
    
    characteristics['recommended_strategy'] = recommended_strategy
    characteristics['recommendation_reason'] = reason
    
    print(f"📈 Dataset Analysis Results:")
    print(f"   Total images: {characteristics['total_images']:,}")
    print(f"   Total categories: {characteristics['total_categories']:,}")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"   Avg categories per image: {avg_categories_per_image:.2f}")
    print(f"   Unique combinations: {characteristics['unique_combinations']:,}")
    print(f"\n💡 Recommended strategy: {recommended_strategy}")
    print(f"   Reason: {reason}")
    
    return characteristics


def validate_split_quality(stats: Dict[str, Dict[str, int]], 
                          min_val_ratio: float = 0.1,
                          min_test_ratio: float = 0.05) -> Dict[str, bool]:
    """분할 품질 검증"""
    quality_check = {
        'adequate_validation_size': False,
        'adequate_test_size': False, 
        'no_empty_splits': False,
        'balanced_distribution': False
    }
    
    total_images = sum(split_data['images'] for split_data in stats.values())
    
    if total_images == 0:
        return quality_check
    
    val_ratio = stats.get('val', {}).get('images', 0) / total_images
    test_ratio = stats.get('test', {}).get('images', 0) / total_images
    
    quality_check['adequate_validation_size'] = val_ratio >= min_val_ratio
    quality_check['adequate_test_size'] = test_ratio >= min_test_ratio
    quality_check['no_empty_splits'] = all(split_data.get('images', 0) > 0 
                                          for split_data in stats.values())
    
    # 분포 균형성 체크 (train이 너무 치우치지 않았는지)
    train_ratio = stats.get('train', {}).get('images', 0) / total_images
    quality_check['balanced_distribution'] = 0.5 <= train_ratio <= 0.9
    
    return quality_check


def _visualize_annotations_on_image(
    image_info: Dict,
    annotations: List[Dict],
    title: str,
    save_path: Path,
    source_image_path: Optional[Path] = None
):
    """이미지 위에 어노테이션을 그리고 저장하는 내부 함수"""
    try:
        # 원본 이미지 경로가 있으면 로드, 없으면 흰 배경 생성
        if source_image_path and source_image_path.exists():
            image = cv2.imread(str(source_image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            width = image_info['width']
            height = image_info['height']
            image = np.ones((height, width, 3), dtype=np.uint8) * 255
            if not source_image_path:
                print(f"      ⚠️ Source image not found for {image_info['file_name']}. Drawing on white background.")

        for ann in annotations:
            bbox = ann['bbox']
            x, y, w, h = [int(c) for c in bbox]
            # 빨간색 사각형으로 바운딩 박스 그리기
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        plt.figure(figsize=(16, 12))
        plt.imshow(image)
        plt.title(f"Distribution of '{title}' in {image_info['file_name']} ({len(annotations)} instances)")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"      ❌ Failed to visualize for {image_info['file_name']}: {e}")

if __name__ == "__main__":
    print("Starting comprehensive dataset splitting examples...")
    
    base_dir = Path(os.getcwd()).resolve()
    
    data_path = base_dir / "assets"
    
    # 데이터셋 특성 분석 및 추천
    # characteristics = analyze_dataset_characteristics(data_path / "merged_dataset.json")
    # recommended_strategy = characteristics['recommended_strategy']

    # 원본 이미지가 있는 디렉토리 목록
    source_directories = [
        data_path / "TS",
        data_path / "VS",
    ]

    # 소수 클래스 분포 시각화 분석
    # run_rare_class_analysis = input("\n🔬 Analyze rare class locality and visualize? (y/n): ").lower().strip() == 'y'
    # if run_rare_class_analysis:
    #     # 분석을 위해 splitter 인스턴스 생성 및 데이터 로드
    #     analysis_splitter = create_splitter(input_json=data_path / "merged_dataset.json", output_dir=data_path / "temp_for_analysis")
    #     analyze_rare_class_locality(
    #         splitter=analysis_splitter,
    #         output_dir=data_path / "rare_class_analysis",
    #         source_dirs=source_directories
    #     )
    
    # 분할
    # print(f"\nUsing recommended strategy: {recommended_strategy}")
    stats = quick_split(
        input_json=data_path / "merged_v01_prepro.json",
        # input_json=data_path / "merged_dataset.json",
        # output_dir=data_path / f"recommended_split_{recommended_strategy}",
        output_dir=data_path / "strategy_comparison_v01" / "random_split",
        # strategy=recommended_strategy,
        strategy="random",
        train_ratio=0.8,
        val_ratio=0.1,
        image_dir=None
    )
    
    # 품질 검증
    # quality = validate_split_quality(stats)
    # print(f"Quality Check: {quality}")
    
    # 여러 전략 비교
    # comparison_results = compare_strategies(
    #     input_json=data_path / "merged_dataset.json",
    #     output_base_dir=data_path / "strategy_comparison"
    # )
    
    
    # # 5. 사용자 맞춤형 설정 예시 (극심한 불균형용)
    # if characteristics['imbalance_ratio'] > 100:
    #     print(f"Extreme imbalance detected ({characteristics['imbalance_ratio']:.1f}:1)")
    #     print("Creating custom split for extreme imbalance...\n")
        
    #     custom_splitter = create_splitter(
    #         input_json=data_path / "merged_dataset.json",
    #         output_dir=data_path / "extreme_custom_split",
    #         strategy="hybrid",
    #         train_ratio=0.8,  # 더 많이 train에 할당
    #         val_ratio=0.15
    #     )
        
    #     custom_splitter.set_strategy_params(
    #         rare_threshold=min(100, max(characteristics['class_distribution'].values()) // 50),
    #         very_rare_threshold=min(20, max(characteristics['class_distribution'].values()) // 200),
    #         min_samples_per_category=2,
    #         min_samples_per_split=1
    #     )
        
    #     custom_stats = custom_splitter.split()
    #     print("Custom extreme imbalance split completed!\n")

    
    
    print(f"All dataset splitting operations completed successfully!")