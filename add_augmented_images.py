"""
Targeted Augmentation Script
Add 100 augmented images to specific categories only
- metal, biological, clothes, plastic, shoes, trash
Makes it fair since these had no augmentation before
"""

import random
from pathlib import Path
import time
from PIL import Image, ImageEnhance

def augment_image(image_path, output_path, augmentation_type):
    """Apply augmentation to an image"""
    try:
        img = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        elif img.mode == 'P':
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
        
        img.save(output_path, quality=95)
        return True
    except Exception as e:
        return False


def add_augmented_images(category_path, category_name, add_count=100):
    """Add augmented images to a category"""
    print(f"\n{category_name:15} ", end='', flush=True)
    
    # Get original images (not augmented)
    all_images = list(Path(category_path).glob('*'))
    original_images = [f for f in all_images 
                      if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif'] 
                      and not f.name.startswith('aug_')]
    
    if not original_images:
        print("✗ No original images")
        return 0
    
    augmentation_types = [
        'rotate_90', 'rotate_270', 'flip_horizontal', 'flip_vertical',
        'brightness_up', 'brightness_down', 'contrast_up', 'contrast_down',
        'color_shift', 'sharpness', 'rotate_15', 'rotate_345'
    ]
    
    generated = 0
    attempts = 0
    max_attempts = add_count * 2
    
    while generated < add_count and attempts < max_attempts:
        source_img = random.choice(original_images)
        aug_type = random.choice(augmentation_types)
        output_name = f"aug_{aug_type}_{int(time.time())}_{attempts}.jpg"
        output_path = Path(category_path) / output_name
        
        if not output_path.exists():
            if augment_image(source_img, output_path, aug_type):
                generated += 1
        
        attempts += 1
        time.sleep(0.01)
    
    print(f"✓ Added {generated} images")
    return generated


def main():
    dataset_path = Path(__file__).parent / 'data' / 'garbage_classification'
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    # Only these 6 categories
    target_categories = ['metal', 'biological', 'clothes', 'plastic', 'shoes', 'trash']
    
    print("="*60)
    print("TARGETED AUGMENTATION - Add 100 images to 6 categories")
    print("="*60)
    print("\nCategories to augment:")
    for cat in target_categories:
        print(f"  ✓ {cat}")
    
    print("\n" + "="*60)
    response = input("Add 100 images to these 6 categories? (yes/no): ").lower()
    if response != 'yes':
        print("Cancelled.")
        return
    
    print("\nAugmenting (100 images per category):\n")
    
    total_added = 0
    for cat_name in target_categories:
        cat_path = dataset_path / cat_name
        if cat_path.exists():
            added = add_augmented_images(cat_path, cat_name, add_count=100)
            total_added += added
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for cat_name in target_categories:
        cat_path = dataset_path / cat_name
        if cat_path.exists():
            count = len([f for f in cat_path.glob('*') 
                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
            print(f"  {cat_name:15} {count} images")
    
    print("\n" + "-"*60)
    print(f"Total added: {total_added} images")
    print("\n✓ Targeted augmentation complete!")


if __name__ == "__main__":
    main()
