"""
Dataset Balancing Script
- Web scrapes images for categories with < 1500 images
- Removes excess images from categories with > 1500 images
- Balances all categories to exactly 1500 images
"""

import os
import shutil
import random
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import hashlib

# Bing Image Search (no API key needed for basic scraping)
def search_bing_images(query, num_images):
    """Search Bing Images and return image URLs"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    images = []
    page = 0
    
    while len(images) < num_images and page < 10:
        url = f"https://www.bing.com/images/search?q={query}&first={page * 35}&count=35"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find image links
            for img in soup.find_all('a', {'class': 'm'}):
                if 'murl' in img.get('m', ''):
                    try:
                        import json
                        img_data = json.loads(img['m'])
                        img_url = img_data.get('murl')
                        if img_url and img_url not in images:
                            images.append(img_url)
                            if len(images) >= num_images:
                                break
                    except:
                        continue
            
            page += 1
            time.sleep(1)  # Be polite
            
        except Exception as e:
            print(f"Error searching page {page}: {e}")
            break
    
    return images[:num_images]


def download_image(url, save_path, timeout=10):
    """Download image from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        
        if response.status_code == 200:
            # Check if it's actually an image
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return True
        return False
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def get_file_hash(filepath):
    """Get MD5 hash of file to detect duplicates"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def remove_duplicates(folder_path):
    """Remove duplicate images based on file hash"""
    seen_hashes = set()
    duplicates = []
    
    for img_file in Path(folder_path).glob('*'):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            try:
                file_hash = get_file_hash(img_file)
                if file_hash in seen_hashes:
                    duplicates.append(img_file)
                else:
                    seen_hashes.add(file_hash)
            except:
                continue
    
    for dup in duplicates:
        dup.unlink()
    
    return len(duplicates)


# Category search queries - VERY SPECIFIC to avoid mixing categories
SEARCH_QUERIES = {
    'battery': 'used battery waste pile recycling AA AAA lithium',
    'biological': 'organic food waste compost kitchen scraps vegetable',
    'brown-glass': 'brown amber glass bottle beer wine waste',
    'cardboard': 'corrugated cardboard box waste recycling brown',
    'clothes': 'used clothing textile fabric waste donation pile',
    'green-glass': 'green glass bottle waste wine recycling',
    'metal': 'aluminum tin metal can waste steel scrap',
    'paper': 'waste paper newspaper magazine recycling pile',
    'plastic': 'plastic bottle PET waste recycling clear container',
    'shoes': 'old worn shoes sneakers boots waste pair',
    'trash': 'mixed garbage trash bin waste landfill',
    'white-glass': 'clear transparent glass bottle waste recycling'
}

# Exclude terms to avoid wrong images
EXCLUDE_TERMS = {
    'battery': ['clothes', 'shoes', 'paper', 'cardboard'],
    'biological': ['plastic', 'metal', 'glass', 'battery'],
    'brown-glass': ['plastic', 'metal', 'paper', 'cardboard', 'clothes'],
    'cardboard': ['plastic', 'metal', 'glass', 'clothes', 'shoes'],
    'clothes': ['plastic', 'bottle', 'glass', 'metal', 'cardboard', 'shoes'],
    'green-glass': ['plastic', 'metal', 'paper', 'cardboard', 'clothes'],
    'metal': ['plastic', 'glass', 'paper', 'cardboard', 'clothes'],
    'paper': ['plastic', 'metal', 'glass', 'cardboard', 'clothes'],
    'plastic': ['glass', 'metal', 'paper', 'cardboard', 'clothes', 'shoes'],
    'shoes': ['clothes', 'plastic', 'bottle', 'glass', 'metal', 'cardboard'],
    'trash': ['bottle', 'can', 'specific'],
    'white-glass': ['plastic', 'metal', 'paper', 'cardboard', 'clothes']
}


def balance_category(category_path, category_name, target_count=1500, verify=True):
    """Balance a single category to target count"""
    print(f"\n{'='*60}")
    print(f"Processing: {category_name}")
    print(f"{'='*60}")
    
    # Get current images
    current_images = list(Path(category_path).glob('*'))
    current_images = [f for f in current_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']]
    current_count = len(current_images)
    
    print(f"Current count: {current_count}")
    
    if current_count == target_count:
        print(f"✓ Already balanced!")
        return
    
    elif current_count > target_count:
        # Remove excess images randomly
        excess = current_count - target_count
        print(f"Removing {excess} excess images...")
        
        # Ask for confirmation before removing
        if verify:
            response = input(f"Remove {excess} images from {category_name}? (y/n): ").lower()
            if response != 'y':
                print("Skipped.")
                return
        
        random.shuffle(current_images)
        for img in current_images[:excess]:
            img.unlink()
        
        print(f"✓ Removed {excess} images. New count: {target_count}")
    
    else:
        # Need to add images
        needed = target_count - current_count
        print(f"Need to add {needed} images")
        
        # Remove duplicates first
        print("Checking for duplicates...")
        dup_count = remove_duplicates(category_path)
        if dup_count > 0:
            print(f"Removed {dup_count} duplicate images")
            current_count -= dup_count
            needed = target_count - current_count
        
        if needed <= 0:
            print(f"✓ After removing duplicates, count is now {current_count}")
            return
        
        # Ask for confirmation before scraping
        if verify:
            response = input(f"Download {needed} images for {category_name}? (y/n): ").lower()
            if response != 'y':
                print("Skipped.")
                return
        
        # Web scrape images with SPECIFIC query
        search_query = SEARCH_QUERIES.get(category_name, f"{category_name} waste recycling")
        exclude_terms = EXCLUDE_TERMS.get(category_name, [])
        
        print(f"Search query: '{search_query}'")
        print(f"Excluding: {', '.join(exclude_terms)}")
        print(f"Looking for {needed} images...")
        
        # Search for more images than needed (some downloads will fail)
        image_urls = search_bing_images(search_query, needed * 3)
        print(f"Found {len(image_urls)} image URLs")
        
        downloaded = 0
        failed = 0
        
        for i, url in enumerate(image_urls):
            if downloaded >= needed:
                break
            
            # Skip URLs with exclude terms
            url_lower = url.lower()
            if any(term in url_lower for term in exclude_terms):
                continue
            
            # Generate unique filename
            ext = '.jpg'
            if '.' in url.split('/')[-1]:
                ext = '.' + url.split('.')[-1].split('?')[0][:4]
            
            filename = f"scraped_{category_name}_{int(time.time())}_{i}{ext}"
            save_path = Path(category_path) / filename
            
            print(f"Downloading {downloaded + 1}/{needed}... (failed: {failed})", end='\r')
            
            if download_image(url, save_path):
                # Verify image is not corrupted
                try:
                    from PIL import Image
                    img = Image.open(save_path)
                    img.verify()
                    downloaded += 1
                except:
                    save_path.unlink()
                    failed += 1
            else:
                failed += 1
            
            time.sleep(0.5)  # Be polite to servers
        
        print(f"\n✓ Downloaded {downloaded} new images (failed: {failed})")
        
        # Remove duplicates again
        dup_count = remove_duplicates(category_path)
        if dup_count > 0:
            print(f"Removed {dup_count} duplicate images after download")
        
        final_count = len([f for f in Path(category_path).glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
        print(f"Final count: {final_count}")
        
        if final_count < target_count:
            print(f"⚠ Warning: Could only get {final_count}/{target_count} images")
            print(f"   You may need to manually add {target_count - final_count} more images")


def main():
    dataset_path = Path(__file__).parent / 'data' / 'garbage_classification'
    
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    print("="*60)
    print("DATASET BALANCING SCRIPT")
    print("="*60)
    print(f"Target: 1500 images per category")
    print(f"Dataset path: {dataset_path}")
    print("\n⚠ IMPORTANT SAFETY FEATURES:")
    print("  - Very specific search queries to avoid mixing categories")
    print("  - URL filtering to exclude wrong terms")
    print("  - Image verification to check validity")
    print("  - Duplicate detection and removal")
    print("  - Manual confirmation before each category")
    
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    categories.sort(key=lambda x: x.name)  # Alphabetical order
    
    print(f"\n📊 CURRENT STATUS:")
    print("-" * 60)
    for cat in categories:
        count = len([f for f in cat.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
        action = "REMOVE" if count > 1500 else "ADD" if count < 1500 else "OK"
        diff = abs(count - 1500)
        print(f"  {cat.name:15} {count:5} images  [{action:6} {diff:4}]")
    
    print("\n" + "="*60)
    print("⚠ WARNING:")
    print("  - Clothes will lose 3,825 images (5325 → 1500)")
    print("  - Shoes will lose 477 images (1977 → 1500)")
    print("  - Other categories will download new images")
    print("="*60)
    
    response = input("\nProceed with balancing? (yes/no): ").lower()
    if response != 'yes':
        print("Cancelled.")
        return
    
    print("\n✓ Starting balanced dataset creation...")
    print("  (You'll be asked to confirm each category)")
    
    for category in categories:
        try:
            balance_category(category, category.name, target_count=1500, verify=True)
        except KeyboardInterrupt:
            print("\n\n⚠ Balancing interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Error processing {category.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    total = 0
    for cat in categories:
        count = len([f for f in cat.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
        status = "✓" if count == 1500 else "⚠"
        print(f"{status} {cat.name:15} {count:5} images")
        total += count
    
    print("-" * 60)
    print(f"   TOTAL:          {total:5} images")
    print("\n✓ Dataset balancing complete!")
    print("\n💡 NEXT STEPS:")
    print("  1. Manually verify scraped images in each folder")
    print("  2. Remove any incorrectly categorized images")
    print("  3. Re-train model with balanced dataset")


if __name__ == "__main__":
    main()
