import json
import random
from pathlib import Path
from collections import defaultdict, Counter
import shutil
from typing import Dict, List, Tuple, Union, Optional
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class SplitStrategy(Enum):
    """분할 전략 열거형"""
    DOMINANT_CATEGORY = "dominant_category"
    MULTI_LABEL = "multi_label"
    HYBRID = "hybrid"


class StratifiedDatasetSplitter:
    """
    COCO 형식 데이터셋을 위한 종합 Stratified Split 클래스
    
    세 가지 전략 지원:
    1. Dominant Category: 이미지당 가장 많은 어노테이션을 가진 카테고리 기준
    2. Multi-label: 모든 카테고리 조합을 고려한 정밀 분할
    3. Hybrid: Dominant + 소수 카테고리 보정
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
        
        print(f"\n🔄 Applying {self.strategy.value} strategy...")
        
        # 전략별 분할 실행
        if self.strategy == SplitStrategy.DOMINANT_CATEGORY:
            splits = self._dominant_category_split()
        elif self.strategy == SplitStrategy.MULTI_LABEL:
            splits = self._multi_label_split()
        elif self.strategy == SplitStrategy.HYBRID:
            splits = self._hybrid_split()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # 결과 저장 및 검증
        stats = self._save_and_validate_splits(splits)
        
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
                recommended = "multi_label"
            
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
        min_split_samples = self.strategy_params['min_samples_per_split']
        
        for tier_name, class_ids in class_tiers.items():
            for class_id in class_ids:
                images = class_to_images.get(class_id, [])
                if not images:
                    continue
                
                n_images = len(images)
                random.shuffle(images)
                
                if tier_name == 'very_rare':
                    # 극소수: 대부분 train, 최소한의 val
                    if n_images >= 3:
                        splits["train"].extend(images[:-1])
                        splits["val"].extend(images[-1:])
                    else:
                        splits["train"].extend(images)
                        
                elif tier_name == 'rare':
                    # 소수: 보수적 분할 (80:15:5)
                    if n_images >= min_split_samples * 3:
                        n_train = max(min_split_samples, int(n_images * 0.8))
                        n_val = max(1, int(n_images * 0.15))
                        
                        splits["train"].extend(images[:n_train])
                        splits["val"].extend(images[n_train:n_train + n_val])
                        splits["test"].extend(images[n_train + n_val:])
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
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        stats = {}
        
        for split_name, split_images in splits.items():
            if not split_images:
                continue
            
            # 해당 split의 어노테이션 수집
            split_annotations = []
            for img in split_images:
                split_annotations.extend(self.img_to_anns[img['id']])
            
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
        self._generate_validation_report(stats)
        
        print(f"\n✅ Split completed! Results saved to: {self.output_dir}")
        
        return stats
    
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
    
    def _generate_validation_report(self, stats):
        """검증 리포트 생성"""
        print(f"\n" + "="*50)
        print(f"📋 STRATIFIED SPLIT VALIDATION ({self.strategy.value.upper()})")
        print("="*50)
        
        category_dict = {cat['id']: cat['name'] for cat in self.categories}
        
        # 전체 통계
        total_images = sum(split_stats['images'] for split_stats in stats.values())
        total_annotations = sum(split_stats['annotations'] for split_stats in stats.values())
        
        print(f"Overall Distribution:")
        for split_name, split_stats in stats.items():
            img_pct = split_stats['images'] / total_images * 100
            ann_pct = split_stats['annotations'] / total_annotations * 100
            print(f"  {split_name:5}: {split_stats['images']:5,} images ({img_pct:5.1f}%), "
                  f"{split_stats['annotations']:6,} annotations ({ann_pct:5.1f}%)")
        
        # 카테고리별 분포 (상위 20개만)
        print(f"\nTop 20 Categories Distribution:")
        print(f"{'Category':<30} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
        print("-" * 70)
        
        # 전체 카테고리를 어노테이션 수 기준으로 정렬
        all_categories = set()
        for split_stats in stats.values():
            all_categories.update(split_stats['categories'].keys())
        
        category_totals = {}
        for cat_id in all_categories:
            total = sum(stats[split]['categories'].get(cat_id, 0) 
                       for split in ['train', 'val', 'test'])
            category_totals[cat_id] = total
        
        # 상위 20개 카테고리 출력
        sorted_categories = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        
        for cat_id, total in sorted_categories[:20]:
            cat_name = category_dict.get(cat_id, f'Unknown_{cat_id}')[:28]
            train_count = stats.get('train', {}).get('categories', {}).get(cat_id, 0)
            val_count = stats.get('val', {}).get('categories', {}).get(cat_id, 0)
            test_count = stats.get('test', {}).get('categories', {}).get(cat_id, 0)
            
            print(f"{cat_name:<30} {train_count:8,} {val_count:8,} {test_count:8,} {total:8,}")
        
        if len(sorted_categories) > 20:
            print(f"... and {len(sorted_categories) - 20} more categories")


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
) -> Dict[str, Dict[str, int]]:
    """원샷 분할 함수"""
    splitter = create_splitter(
        input_json=input_json,
        output_dir=output_dir,
        strategy=strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        **kwargs
    )
    
    return splitter.split()


# 추가 유틸리티 함수들
def compare_strategies(
    input_json: Union[str, Path],
    output_base_dir: Union[str, Path] = "strategy_comparison",
    strategies: List[str] = None
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """여러 전략을 비교하는 함수"""
    if strategies is None:
        strategies = ["dominant_category", "multi_label", "hybrid"]
    
    results = {}
    output_base = Path(output_base_dir)
    
    for strategy in strategies:
        print(f"\n🔄 Comparing strategy: {strategy}")
        try:
            stats = quick_split(
                input_json=input_json,
                output_dir=output_base / strategy,
                strategy=strategy
            )
            results[strategy] = stats
            
        except Exception as e:
            print(f"❌ Error with {strategy}: {e}")
            results[strategy] = None
    
    # 비교 리포트 생성
    _generate_comparison_report(results)
    return results


def _generate_comparison_report(results):
    """전략 비교 리포트 생성"""
    print(f"\n" + "="*60)
    print("📊 STRATEGY COMPARISON REPORT")
    print("="*60)
    
    print(f"{'Strategy':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total Ann.':<12}")
    print("-" * 60)
    
    for strategy, stats in results.items():
        if stats is None:
            print(f"{strategy:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} {'ERROR':<12}")
            continue
            
        train_imgs = stats.get('train', {}).get('images', 0)
        val_imgs = stats.get('val', {}).get('images', 0) 
        test_imgs = stats.get('test', {}).get('images', 0)
        total_ann = sum(split_data.get('annotations', 0) for split_data in stats.values())
        
        print(f"{strategy:<20} {train_imgs:<8,} {val_imgs:<8,} {test_imgs:<8,} {total_ann:<12,}")


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
        recommended_strategy = "multi_label"
        reason = "High imbalance, precise distribution needed"
    
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


if __name__ == "__main__":
    # 🎯 실제 사용 예시들
    
    print("🚀 Starting comprehensive dataset splitting examples...")
    
    # 1. 데이터셋 특성 분석 및 추천
    characteristics = analyze_dataset_characteristics("assets/merged_dataset.json")
    recommended_strategy = characteristics['recommended_strategy']
    
    # 2. 추천 전략으로 분할
    print(f"\n🎯 Using recommended strategy: {recommended_strategy}")
    stats = quick_split(
        input_json="assets/merged_dataset.json",
        output_dir=f"assets/recommended_split_{recommended_strategy}",
        strategy=recommended_strategy,
        image_dir="assets/images"
    )
    
    # 3. 품질 검증
    quality = validate_split_quality(stats)
    print(f"\n✅ Quality Check: {quality}")
    
    # 4. 여러 전략 비교 (선택적)
    compare_all = input("\n🤔 Compare all strategies? (y/n): ").lower().strip() == 'y'
    if compare_all:
        comparison_results = compare_strategies(
            input_json="assets/merged_dataset.json",
            output_base_dir="assets/strategy_comparison"
        )
    
    print("\n🎉 All examples completed!")
    
    # 5. 사용자 맞춤형 설정 예시 (극심한 불균형용)
    if characteristics['imbalance_ratio'] > 100:
        print(f"\nExtreme imbalance detected ({characteristics['imbalance_ratio']:.1f}:1)")
        print("🔧 Creating custom split for extreme imbalance...")
        
        custom_splitter = create_splitter(
            input_json="assets/merged_dataset.json",
            output_dir="assets/extreme_custom_split",
            strategy="hybrid",
            train_ratio=0.8,  # 더 많이 train에 할당
            val_ratio=0.15
        )
        
        custom_splitter.set_strategy_params(
            rare_threshold=min(100, max(characteristics['class_distribution'].values()) // 50),
            very_rare_threshold=min(20, max(characteristics['class_distribution'].values()) // 200),
            min_samples_per_category=2,
            min_samples_per_split=1
        )
        
        custom_stats = custom_splitter.split()
        print("✅ Custom extreme imbalance split completed!")
    
    print(f"\n🏁 All dataset splitting operations completed successfully!")