import os
import shutil
from pathlib import Path
from typing import Union, List, Dict, Optional
import json
from collections import defaultdict
import glob

class ImagePathManager:
    """
    이미지 경로 관리를 위한 하이브리드 솔루션
    
    두 가지 모드 지원:
    1. COPY: 이미지를 split 디렉토리로 복사 (권장)
    2. SYMLINK: 심볼릭 링크 생성 (저장공간 절약)
    3. PATH_MAPPING: JSON에 원본 경로 저장 (최소 공간)
    """
    
    def __init__(
        self,
        source_dirs: List[Union[str, Path]],
        output_dir: Union[str, Path],
        mode: str = "copy"  # "copy", "symlink", "path_mapping"
    ):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.output_dir = Path(output_dir)
        self.mode = mode.lower()
        
        # 이미지 파일 확장자
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    def find_all_images(self) -> Dict[str, Path]:
        """모든 이미지 파일을 찾아서 파일명 -> 전체경로 매핑 생성"""
        image_mapping = {}
        
        print(f"🔍 Scanning for images in {len(self.source_dirs)} directories...")
        
        for source_dir in self.source_dirs:
            if not source_dir.exists():
                print(f"⚠️  Directory not found: {source_dir}")
                continue
                
            # 재귀적으로 이미지 파일 찾기
            for pattern in ['**/*.jpg', '**/*.jpeg', '**/*.png', '**/*.bmp']:
                for img_path in source_dir.glob(pattern):
                    filename = img_path.name
                    
                    if filename in image_mapping:
                        print(f"⚠️  Duplicate filename found: {filename}")
                        print(f"     Existing: {image_mapping[filename]}")
                        print(f"     New: {img_path}")
                        # 더 최근 파일을 선택하거나, 사용자 정의 로직 적용
                        continue
                    
                    image_mapping[filename] = img_path
        
        print(f"✅ Found {len(image_mapping):,} unique images")
        return image_mapping
    
    def setup_split_directories(self, splits: List[str] = None):
        """Split 디렉토리 구조 생성"""
        if splits is None:
            splits = ['train', 'val', 'test']
        
        for split in splits:
            split_dir = self.output_dir / split
            if self.mode in ['copy', 'symlink']:
                (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            else:  # path_mapping
                split_dir.mkdir(parents=True, exist_ok=True)
    
    def organize_images_for_splits(
        self,
        split_data: Dict[str, List[Dict]],  # split_name -> list of image info
        image_mapping: Dict[str, Path]
    ) -> Dict[str, List[str]]:
        """
        Split별로 이미지 구성
        
        Args:
            split_data: {'train': [{'file_name': 'img1.jpg', ...}], ...}
            image_mapping: {'img1.jpg': Path('/full/path/img1.jpg'), ...}
            
        Returns:
            {'train': ['missing_files_list'], ...}  # 누락된 파일 목록
        """
        missing_files = defaultdict(list)
        
        self.setup_split_directories(list(split_data.keys()))
        
        for split_name, images in split_data.items():
            print(f"\n📁 Organizing {split_name} split ({len(images):,} images)...")
            
            split_dir = self.output_dir / split_name
            copied_count = 0
            
            for img_info in images:
                filename = img_info['file_name']
                
                if filename not in image_mapping:
                    missing_files[split_name].append(filename)
                    continue
                
                src_path = image_mapping[filename]
                
                if self.mode == 'copy':
                    dst_path = split_dir / 'images' / filename
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"❌ Failed to copy {filename}: {e}")
                        missing_files[split_name].append(filename)
                
                elif self.mode == 'symlink':
                    dst_path = split_dir / 'images' / filename
                    try:
                        if dst_path.exists() or dst_path.is_symlink():
                            dst_path.unlink()
                        dst_path.symlink_to(src_path.absolute())
                        copied_count += 1
                    except Exception as e:
                        print(f"❌ Failed to symlink {filename}: {e}")
                        missing_files[split_name].append(filename)
                
                # path_mapping 모드에서는 JSON에만 경로 저장 (실제 파일 이동 없음)
            
            if self.mode in ['copy', 'symlink']:
                print(f"   ✅ {copied_count:,}/{len(images):,} images processed")
                if missing_files[split_name]:
                    print(f"   ⚠️  {len(missing_files[split_name]):,} files missing")
        
        return dict(missing_files)
    
    def create_path_mapping_json(
        self,
        split_data: Dict[str, List[Dict]],
        image_mapping: Dict[str, Path],
        output_file: Union[str, Path] = None
    ):
        """Path mapping 모드용 - 원본 경로를 JSON에 저장"""
        if output_file is None:
            output_file = self.output_dir / "image_path_mapping.json"
        
        path_mapping = {}
        
        for split_name, images in split_data.items():
            path_mapping[split_name] = {}
            
            for img_info in images:
                filename = img_info['file_name']
                if filename in image_mapping:
                    path_mapping[split_name][filename] = str(image_mapping[filename])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(path_mapping, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Path mapping saved to: {output_file}")
        return path_mapping


def create_custom_dataset_loader_code():
    """Path mapping 모드용 커스텀 데이터로더 코드 생성"""
    code = '''
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path

class PathMappingDataset(Dataset):
    """원본 경로에서 직접 로드하는 데이터셋"""
    
    def __init__(self, annotations_file, path_mapping_file, transform=None):
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        with open(path_mapping_file, 'r') as f:
            all_mappings = json.load(f)
        
        # 현재 split의 경로 매핑만 추출
        split_name = Path(annotations_file).stem  # train.json -> train
        self.path_mapping = all_mappings.get(split_name, {})
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations'] 
        self.transform = transform
        
        # 이미지별 어노테이션 매핑
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_info = self.images[idx]
        filename = img_info['file_name']
        img_id = img_info['id']
        
        # 원본 경로에서 이미지 로드
        if filename in self.path_mapping:
            img_path = self.path_mapping[filename]
            image = Image.open(img_path).convert('RGB')
        else:
            raise FileNotFoundError(f"Image not found: {filename}")
        
        # 어노테이션 가져오기
        annotations = self.img_to_anns.get(img_id, [])
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'annotations': annotations,
            'image_id': img_id,
            'filename': filename
        }

# 사용 예시
# dataset = PathMappingDataset(
#     annotations_file='annotations/train.json',
#     path_mapping_file='image_path_mapping.json',
#     transform=your_transforms
# )
'''
    return code


# 통합 이미지 관리 함수
def organize_dataset_images(
    coco_splits: Dict[str, Dict],  # {'train': coco_data, 'val': coco_data, ...}
    source_dirs: List[Union[str, Path]],
    output_dir: Union[str, Path],
    mode: str = "copy",  # "copy", "symlink", "path_mapping"
    save_custom_loader: bool = True
) -> Dict[str, List[str]]:
    """
    COCO 분할 데이터의 이미지들을 효율적으로 구성
    
    Args:
        coco_splits: Split별 COCO 데이터 {'train': {...}, 'val': {...}, ...}
        source_dirs: 원본 이미지가 있는 디렉토리들
        output_dir: 출력 디렉토리
        mode: 처리 모드 ("copy", "symlink", "path_mapping")
        save_custom_loader: path_mapping 모드시 커스텀 로더 코드 저장 여부
    
    Returns:
        누락된 파일 목록 {'split_name': ['missing_file1', ...], ...}
    """
    
    # 이미지 파일 스캔
    manager = ImagePathManager(source_dirs, output_dir, mode)
    image_mapping = manager.find_all_images()
    
    # Split별 이미지 정보 추출
    split_image_data = {}
    for split_name, coco_data in coco_splits.items():
        split_image_data[split_name] = coco_data.get('images', [])
    
    # 이미지 구성
    missing_files = manager.organize_images_for_splits(split_image_data, image_mapping)
    
    # Path mapping 모드인 경우 매핑 파일 생성
    if mode == 'path_mapping':
        manager.create_path_mapping_json(split_image_data, image_mapping)
        
        if save_custom_loader:
            loader_code = create_custom_dataset_loader_code()
            with open(Path(output_dir) / 'custom_dataset_loader.py', 'w') as f:
                f.write(loader_code)
            print("📝 Custom dataset loader saved to: custom_dataset_loader.py")
    
    return missing_files


# 편의 함수들
def quick_copy_images(coco_splits: Dict[str, Dict], source_dirs: List[str | Path], output_dir: str | Path):
    """빠른 이미지 복사"""
    return organize_dataset_images(coco_splits, source_dirs, output_dir, mode="copy")

def quick_symlink_images(coco_splits: Dict[str, Dict], source_dirs: List[str | Path], output_dir: str | Path):  
    """빠른 심볼릭 링크 생성"""
    return organize_dataset_images(coco_splits, source_dirs, output_dir, mode="symlink")

def quick_path_mapping(coco_splits: Dict[str, Dict], source_dirs: List[str | Path], output_dir: str | Path):
    """빠른 경로 매핑 생성"""  
    return organize_dataset_images(coco_splits, source_dirs, output_dir, mode="path_mapping")


if __name__ == "__main__":
    
    base_dir = Path(os.getcwd()).resolve()
    
    data_path = base_dir / "assets"
    
    # 1. 병합된 전체 COCO 데이터 로드
    with open(data_path / "merged_dataset.json", 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    # organize_dataset_images 함수 형식에 맞게 데이터를 래핑
    # 전체 데이터를 'all'이라는 이름의 단일 스플릿으로 취급
    splits_for_organizing = {'all': merged_data}
    
    # 2. 원본 이미지 디렉토리들
    source_directories = [
        data_path / "TS",
        data_path / "VS",
    ]
    
    # 3-A. 복사 모드 (권장)
    print("🚀 Mode 1: Copying images...")
    missing_copy = quick_copy_images(
        coco_splits=splits_for_organizing,
        source_dirs=source_directories, 
        output_dir=data_path / "image"
    )
    
    # 3-B. 심볼릭 링크 모드 (저장공간 절약)
    # print("\n🔗 Mode 2: Creating symbolic links...")
    # missing_symlink = quick_symlink_images(
    #     coco_splits=splits_for_organizing,
    #     source_dirs=source_directories,
    #     output_dir=data_path / "image_symlink" 
    # )
    
    # 3-C. 경로 매핑 모드 (최소 공간)
    # print("\n📄 Mode 3: Creating path mapping...")
    # missing_mapping = quick_path_mapping(
    #     coco_splits=splits_for_organizing,
    #     source_dirs=source_directories,
    #     output_dir=data_path / "image_path_mapping"
    # )
    
    print(f"\n📊 Results Summary:")
    print(f"   Copy mode missing files: {sum(len(v) for v in missing_copy.values())}")
    # print(f"   Symlink mode missing files: {sum(len(v) for v in missing_symlink.values())}")
    # print(f"   Path mapping missing files: {sum(len(v) for v in missing_mapping.values())}")