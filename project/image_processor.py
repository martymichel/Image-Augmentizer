from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import List
from augmentation_types import AugmentationConfig, AugmentationType

class ImageProcessor(QThread):
    progress_signal = pyqtSignal(int, int, str)
    finished_signal = pyqtSignal(dict)
    
    def __init__(self, input_dir: Path, configs: List[AugmentationConfig]):
        super().__init__()
        self.input_dir = input_dir
        self.configs = configs
        self.output_dir = input_dir.parent / 'augmented_images'
    
    def run(self):
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Collect all images
        image_files = []
        original_counts = {}
        for subdir in ['OK', 'NOK', 'background']:
            subdir_path = self.input_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob('*.jpg')) + \
                        list(subdir_path.glob('*.png'))
                image_files.extend(files)
                original_counts[subdir] = len(files)
        
        total_files = len(image_files)
        processed = 0
        augmented_counts = {'OK': 0, 'NOK': 0, 'background': 0}
        
        # Process each image
        for img_path in image_files:
            self.progress_signal.emit(
                processed, total_files,
                f"Processing {img_path.name}..."
            )
            
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Create output subdirectory
            out_subdir = self.output_dir / img_path.parent.name
            out_subdir.mkdir(exist_ok=True)
            
            # Generate augmented versions
            augmented = self.generate_augmentations(img)
            
            # Save augmented images
            for i, aug_img in enumerate(augmented):
                out_path = out_subdir / f"{img_path.stem}_aug_{i}{img_path.suffix}"
                cv2.imwrite(str(out_path), aug_img)
                augmented_counts[img_path.parent.name] += 1
            
            processed += 1
        
        # Emit completion signal with summary
        self.finished_signal.emit({
            'original_counts': original_counts,
            'augmented_counts': augmented_counts,
            'total_images': sum(original_counts.values()) + 
                          sum(augmented_counts.values())
        })
    
    def generate_augmentations(self, img: np.ndarray) -> List[np.ndarray]:
        height, width = img.shape[:2]
        augmented = []
        combinations = []
        
        # Generate all combinations of augmentation levels
        for level1 in range(3):  # 0=none, 1=min, 2=max
            for level2 in range(3):
                for level3 in range(3):
                    if level1 == level2 == level3 == 0:
                        continue  # Skip no-augmentation case
                    combinations.append((level1, level2, level3))
        
        # Process each combination
        for i, (level1, level2, level3) in enumerate(combinations):
            result = img.copy()
            
            # Apply each augmentation
            for config, level in zip(self.configs, [level1, level2, level3]):
                if level == 0:
                    continue
                    
                # Calculate intensity for this level
                intensity = (
                    config.min_value if level == 1 
                    else config.max_value
                )
                
                # Apply the specific augmentation
                if config.type == AugmentationType.SHIFT_X:
                    shift = int(width * intensity)
                    matrix = np.float32([[1, 0, shift], [0, 1, 0]])
                    result = cv2.warpAffine(
                        result, matrix, (width, height),
                        borderMode=cv2.BORDER_REFLECT
                    )
                    
                elif config.type == AugmentationType.SHIFT_Y:
                    shift = int(height * intensity)
                    matrix = np.float32([[1, 0, 0], [0, 1, shift]])
                    result = cv2.warpAffine(
                        result, matrix, (width, height),
                        borderMode=cv2.BORDER_REFLECT
                    )
                    
                elif config.type == AugmentationType.ROTATION:
                    angle = 360 * intensity
                    matrix = cv2.getRotationMatrix2D(
                        (width/2, height/2), angle, 1.0
                    )
                    result = cv2.warpAffine(
                        result, matrix, (width, height),
                        borderMode=cv2.BORDER_REFLECT
                    )
                    
                elif config.type == AugmentationType.BRIGHTNESS:
                    result = cv2.convertScaleAbs(
                        result, alpha=1.0, beta=255*intensity
                    )
                    
                elif config.type == AugmentationType.CONTRAST:
                    result = cv2.convertScaleAbs(
                        result, alpha=1.0 + intensity
                    )
                    
                elif config.type == AugmentationType.BLUR:
                    k_size = 1 + 2 * int(10 * intensity)
                    result = cv2.GaussianBlur(
                        result, (k_size, k_size), 0
                    )
                    
                elif config.type == AugmentationType.NOISE:
                    noise = np.random.normal(
                        0, 255 * intensity, result.shape
                    ).astype(np.uint8)
                    result = cv2.add(result, noise)
                    
                elif config.type == AugmentationType.ZOOM:
                    scale = 1.0 + intensity
                    matrix = cv2.getRotationMatrix2D(
                        (width/2, height/2), 0, scale
                    )
                    result = cv2.warpAffine(
                        result, matrix, (width, height),
                        borderMode=cv2.BORDER_REFLECT
                    )
            
            # Save augmented image with combination index
            augmented.append(result)
        
        return augmented