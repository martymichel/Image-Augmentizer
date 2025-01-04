import os
import cv2
import numpy as np
from pathlib import Path

class ImageAugmenter:
    def __init__(self):
        # Define augmentation parameters for each level
        self.augmentations = {
            'brightness': {
                0: 1.0,    # Original
                1: 1.2,    # Slight brightness increase
                2: 1.5     # Strong brightness increase
            },
            'rotation': {
                0: 0,      # Original
                1: 15,     # 15 degrees
                2: 30      # 30 degrees
            },
            'blur': {
                0: 0,      # Original
                1: 3,      # Slight blur
                2: 7       # Strong blur
            }
        }

    def apply_brightness(self, image, level):
        factor = self.augmentations['brightness'][level]
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def apply_rotation(self, image, level):
        angle = self.augmentations['rotation'][level]
        if angle == 0:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height), 
                            borderMode=cv2.BORDER_REFLECT)

    def apply_blur(self, image, level):
        kernel_size = self.augmentations['blur'][level]
        if kernel_size == 0:
            return image
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def process_image(self, image_path, output_dir):
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Create output directory structure
            relative_path = image_path.parent.name
            output_subdir = output_dir / relative_path
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Generate all combinations of augmentation levels
            for aug1 in range(3):  # brightness
                for aug2 in range(3):  # rotation
                    for aug3 in range(3):  # blur
                        # Skip if all augmentations are 0 (original image)
                        if aug1 == 0 and aug2 == 0 and aug3 == 0:
                            continue

                        # Apply augmentations sequentially
                        result = image.copy()
                        result = self.apply_brightness(result, aug1)
                        result = self.apply_rotation(result, aug2)
                        result = self.apply_blur(result, aug3)

                        # Generate output filename
                        base_name = image_path.stem
                        extension = image_path.suffix
                        aug_name = f"{base_name}_aug_{aug1}{aug2}{aug3}{extension}"
                        output_path = output_subdir / aug_name

                        # Save augmented image
                        cv2.imwrite(str(output_path), result)

            return True

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return False

def main():
    # Define input and output directories
    input_dir = Path("input")
    output_dir = Path("Augmentiert")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Initialize augmenter
    augmenter = ImageAugmenter()

    # Process all images in subdirectories
    subdirs = [subdir.name for subdir in input_dir.iterdir() if subdir.is_dir()]
    total_processed = 0
    total_failed = 0

    for subdir in subdirs:
        input_subdir = input_dir / subdir
        if not input_subdir.exists():
            print(f"Warning: Directory {input_subdir} does not exist")
            continue

        for image_path in input_subdir.glob('*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                if augmenter.process_image(image_path, output_dir):
                    total_processed += 1
                else:
                    total_failed += 1

    print(f"\nAugmentation complete:")
    print(f"Successfully processed: {total_processed} images")
    print(f"Failed: {total_failed} images")

if __name__ == "__main__":
    main()