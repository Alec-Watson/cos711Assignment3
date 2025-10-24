"""
Data Augmentation Script for Rare Class Images
Applies various augmentation techniques to increase dataset size
"""

import os
from pathlib import Path
import cv2
import albumentations as A
from tqdm import tqdm


INPUT_DIR = "exo/exo_PNG"
OUTPUT_DIR = "exo_augmented"
AUGMENTATIONS_PER_IMAGE = 10

augmentation_pipeline = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=1),
        A.Rotate(limit=30, p=1),
        A.RandomRotate90(p=1),
    ], p=0.8),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1),
        A.GaussNoise(var_limit=(10, 50), p=1),
        A.MotionBlur(blur_limit=5, p=1),
    ], p=0.3),
    
    A.OneOf([
        A.ElasticTransform(alpha=1, sigma=50, p=1),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
    ], p=0.3),
    
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
])


def augment_images():
    """Apply augmentation to all images in input directory"""
    
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    input_path = Path(INPUT_DIR)
    image_files = list(input_path.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {INPUT_DIR}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Generating {AUGMENTATIONS_PER_IMAGE} augmentations per image")
    print(f"Total output: {len(image_files) * (AUGMENTATIONS_PER_IMAGE + 1)} images")
    
    for img_file in tqdm(image_files, desc="Copying originals"):
        img = cv2.imread(str(img_file))
        cv2.imwrite(str(output_path / img_file.name), img)
    
    for img_file in tqdm(image_files, desc="Augmenting"):
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        base_name = img_file.stem
        
        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = augmentation_pipeline(image=img)['image']
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            output_name = f"{base_name}_aug_{i+1}.png"
            cv2.imwrite(str(output_path / output_name), augmented_bgr)
    
    print(f"\n✓ Augmentation complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total images generated: {len(list(output_path.glob('*.png')))}")


if __name__ == "__main__":
    augment_images()

