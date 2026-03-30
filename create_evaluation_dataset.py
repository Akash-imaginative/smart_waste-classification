"""
Create Evaluation Dataset for YOLO
===================================
This script samples images from waste category folders and prepares
a structure for YOLO evaluation (with empty labels folder for future annotation).
"""

import os
import shutil
import random
from pathlib import Path

def create_evaluation_dataset(source_dir='data/val', output_dir='Evaluation', images_per_class=10):
    """
    Sample images from each waste category and create evaluation dataset structure.
    
    Args:
        source_dir: Directory containing category folders (default: data/val)
        output_dir: Output directory for evaluation dataset (default: Evaluation)
        images_per_class: Number of images to sample per category (default: 10)
    """
    
    print("\n" + "="*70)
    print("📂 CREATING EVALUATION DATASET")
    print("="*70 + "\n")
    
    # Define waste categories
    categories = [
        'battery', 'biological', 'brown-glass', 'cardboard', 
        'clothes', 'green-glass', 'metal', 'paper', 
        'plastic', 'shoes', 'trash', 'white-glass'
    ]
    
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"✅ Created directories:")
    print(f"   📁 {images_dir}")
    print(f"   📁 {labels_dir} (empty - ready for annotations)\n")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"❌ Error: Source directory '{source_dir}' not found!")
        return
    
    total_copied = 0
    summary = []
    
    # Process each category
    for category in categories:
        category_path = os.path.join(source_dir, category)
        
        if not os.path.exists(category_path):
            print(f"⚠️  Warning: Category '{category}' not found at {category_path}")
            continue
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.jfif']
        all_images = [
            f for f in os.listdir(category_path)
            if os.path.isfile(os.path.join(category_path, f)) and
            any(f.lower().endswith(ext) for ext in image_extensions)
        ]
        
        if len(all_images) == 0:
            print(f"⚠️  Warning: No images found in {category}")
            continue
        
        # Randomly sample images
        num_to_sample = min(images_per_class, len(all_images))
        selected_images = random.sample(all_images, num_to_sample)
        
        # Copy selected images
        copied_count = 0
        for img_file in selected_images:
            src_path = os.path.join(category_path, img_file)
            # Rename with category prefix to avoid name conflicts
            new_name = f"{category}_{img_file}"
            dst_path = os.path.join(images_dir, new_name)
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                print(f"   ❌ Failed to copy {img_file}: {e}")
        
        total_copied += copied_count
        summary.append((category, copied_count, len(all_images)))
        print(f"✅ {category:15s} → {copied_count:2d}/{images_per_class} images sampled (from {len(all_images)} total)")
    
    # Print summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Total images copied: {total_copied}")
    print(f"Output location: {os.path.abspath(output_dir)}")
    print(f"\n📁 Structure:")
    print(f"   {output_dir}/")
    print(f"   ├── images/     ({total_copied} images)")
    print(f"   └── labels/     (empty - ready for YOLO annotations)")
    print("\n💡 Next steps:")
    print("   1. Annotate images in 'images/' folder using labeling tools")
    print("   2. Save YOLO format labels (.txt) in 'labels/' folder")
    print("   3. Run YOLO validation with this dataset")
    print("="*70 + "\n")


if __name__ == "__main__":
    # You can change these parameters:
    # - source_dir: 'data/val', 'data/train', or 'data/test'
    # - images_per_class: number of images to sample from each category
    
    create_evaluation_dataset(
        source_dir='data/val',      # Change this if needed
        output_dir='Evaluation',
        images_per_class=10
    )
