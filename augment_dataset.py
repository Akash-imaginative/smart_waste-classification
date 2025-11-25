"""
Data Augmentation Script
Safely generates new images from existing ones using transformations
- Rotation, flipping, brightness, contrast, zoom
- 100% safe - no category mixing
- Creates variations to balance dataset to 1500 per category
"""

import os
import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import time

def augment_image(image_path, output_path, augmentation_type):
    """Apply augmentation to an image"""
    try:
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        if augmentation_type == 'rotate_90':
            img = img.rotate(90, expand=True)
        
        elif augmentation_type == 'rotate_270':
            img = img.rotate(270, expand=True)
        
        elif augmentation_type == 'flip_horizontal':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif augmentation_type == 'flip_vertical':
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        
        elif augmentation_type == 'brightness_up':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.3)
        
        elif augmentation_type == 'brightness_down':
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.7)
        
        elif augmentation_type == 'contrast_up':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
        
        elif augmentation_type == 'contrast_down':
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.7)
        
        elif augmentation_type == 'color_shift':
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.2)
        
        elif augmentation_type == 'sharpness':
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)
        
        elif augmentation_type == 'rotate_15':
            img = img.rotate(15, expand=True, fillcolor=(255, 255, 255))
        
        elif augmentation_type == 'rotate_345':
            img = img.rotate(-15, expand=True, fillcolor=(255, 255, 255))
        
        # Save augmented image
        img.save(output_path, quality=95)
        return True
        
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")
        return False


def augment_category(category_path, category_name, target_count=1500):
    """Augment images in a category to reach target count"""
    print(f"\n{'='*60}")
    print(f"Augmenting: {category_name}")
    print(f"{'='*60}")
    
    # Get current images (only original ones, not augmented)
    all_images = list(Path(category_path).glob('*'))
    original_images = [f for f in all_images 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] 
                      and not f.name.startswith('aug_')]
    
    current_count = len(all_images)
    original_count = len(original_images)
    
    print(f"Current total: {current_count} images")
    print(f"Original images: {original_count}")
    
    if current_count >= target_count:
        print(f"✓ Already at target!")
        return
    
    needed = target_count - current_count
    print(f"Need to generate: {needed} images")
    
    if original_count == 0:
        print("✗ No original images to augment!")
        return
    
    # Augmentation techniques
    augmentation_types = [
        'rotate_90', 'rotate_270', 'flip_horizontal', 'flip_vertical',
        'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
        'color_shift', 'sharpness', 'rotate_15', 'rotate_345'
    ]
    
    generated = 0
    attempts = 0
    max_attempts = needed * 2
    
    while generated < needed and attempts < max_attempts:
        # Pick random original image
        source_img = random.choice(original_images)
        
        # Pick random augmentation
        aug_type = random.choice(augmentation_types)
        
        # Generate unique filename
        output_name = f"aug_{aug_type}_{int(time.time())}_{attempts}.jpg"
        output_path = Path(category_path) / output_name
        
        # Check if file already exists
        if output_path.exists():
            attempts += 1
            continue
        
        print(f"Generating {generated + 1}/{needed}... ({aug_type})", end='\r')
        
        if augment_image(source_img, output_path, aug_type):
            generated += 1
        
        attempts += 1
        time.sleep(0.01)  # Small delay to ensure unique timestamps
    
    print(f"\n✓ Generated {generated} augmented images")
    
    final_count = len([f for f in Path(category_path).glob('*') 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
    print(f"Final count: {final_count}/{target_count}")


def main():
    dataset_path = Path(__file__).parent / 'data' / 'garbage_classification'
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    print("="*60)
    print("DATA AUGMENTATION SCRIPT")
    print("="*60)
    print("Target: 1500 images per category")
    print("Method: Generate variations from existing images")
    print("\n✓ Safe - no category mixing possible")
    print("✓ Fast - processes locally")
    print("✓ Quality - professional ML technique")
    
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    categories.sort(key=lambda x: x.name)
    
    print(f"\n📊 CATEGORIES TO AUGMENT:")
    print("-" * 60)
    
    to_augment = []
    for cat in categories:
        count = len([f for f in cat.glob('*') 
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
        if count < 1500:
            needed = 1500 - count
            print(f"  {cat.name:15} {count:5} → 1500  (add {needed})")
            to_augment.append(cat)
        else:
            print(f"  {cat.name:15} {count:5} ✓")
    
    if not to_augment:
        print("\n✓ All categories already balanced!")
        return
    
    print("\n" + "="*60)
    response = input(f"\nAugment {len(to_augment)} categories? (yes/no): ").lower()
    if response != 'yes':
        print("Cancelled.")
        return
    
    print("\n✓ Starting augmentation...\n")
    
    for category in to_augment:
        try:
            augment_category(category, category.name, target_count=1500)
        except KeyboardInterrupt:
            print("\n\n⚠ Augmentation interrupted")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    total = 0
    for cat in categories:
        count = len([f for f in cat.glob('*') 
                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
        status = "✓" if count >= 1500 else "⚠"
        print(f"{status} {cat.name:15} {count:5} images")
        total += count
    
    print("-" * 60)
    print(f"   TOTAL:          {total:5} images")
    print("\n✓ Dataset augmentation complete!")
    print("\n💡 NEXT STEP: Re-train model with balanced dataset")


if __name__ == "__main__":
    main()
