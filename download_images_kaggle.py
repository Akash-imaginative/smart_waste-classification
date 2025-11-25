"""
Alternative: Download images from open datasets
Uses Google Open Images, Kaggle datasets, and other sources
"""

import os
from pathlib import Path

def show_manual_instructions():
    """Show instructions for manually balancing the dataset"""
    
    print("="*70)
    print("MANUAL DATASET BALANCING GUIDE")
    print("="*70)
    
    print("\n📊 Current Status:")
    categories_needed = {
        'battery': 525,
        'brown-glass': 869,
        'cardboard': 148,
        'green-glass': 877,
        'paper': 450,
        'white-glass': 350
    }
    
    for cat, needed in categories_needed.items():
        print(f"  {cat:15} needs {needed:4} more images")
    
    print("\n" + "="*70)
    print("📥 RECOMMENDED SOURCES:")
    print("="*70)
    
    sources = {
        'battery': [
            'https://www.kaggle.com/datasets/search?q=battery+waste',
            'https://pixabay.com/images/search/battery%20waste/',
            'Search: "used batteries waste recycling"'
        ],
        'brown-glass': [
            'https://www.kaggle.com/datasets/search?q=glass+bottle',
            'https://pixabay.com/images/search/brown%20glass%20bottle/',
            'Search: "brown beer wine bottle waste"'
        ],
        'cardboard': [
            'https://www.kaggle.com/datasets/search?q=cardboard+waste',
            'https://pixabay.com/images/search/cardboard%20box/',
            'Search: "cardboard box recycling waste"'
        ],
        'green-glass': [
            'https://www.kaggle.com/datasets/search?q=glass+bottle',
            'https://pixabay.com/images/search/green%20glass%20bottle/',
            'Search: "green wine bottle waste"'
        ],
        'paper': [
            'https://www.kaggle.com/datasets/search?q=paper+waste',
            'https://pixabay.com/images/search/waste%20paper/',
            'Search: "waste paper newspaper recycling"'
        ],
        'white-glass': [
            'https://www.kaggle.com/datasets/search?q=glass+bottle',
            'https://pixabay.com/images/search/clear%20glass%20bottle/',
            'Search: "clear transparent glass bottle waste"'
        ]
    }
    
    print("\n📌 For each category, visit these sources:\n")
    for cat, links in sources.items():
        print(f"\n{cat.upper()} ({categories_needed[cat]} images needed):")
        for i, link in enumerate(links, 1):
            print(f"  {i}. {link}")
    
    print("\n" + "="*70)
    print("🔧 ALTERNATIVE: Use ImageNet or COCO datasets")
    print("="*70)
    print("""
1. Download Google's Open Images dataset:
   https://storage.googleapis.com/openimages/web/index.html
   
2. Use Roboflow public datasets:
   https://universe.roboflow.com/browse/waste-management
   
3. Use existing waste classification datasets:
   - TrashNet: https://github.com/garythung/trashnet
   - Waste Classification Data: https://www.kaggle.com/datasets/techsash/waste-classification-data
   
4. Or manually collect from:
   - Pixabay (free images): https://pixabay.com
   - Unsplash (free images): https://unsplash.com
   - Pexels (free images): https://www.pexels.com
    """)
    
    print("="*70)
    print("💡 QUICK FIX: Data Augmentation Instead!")
    print("="*70)
    print("""
Instead of downloading new images, we can AUGMENT existing images:
- Rotation, flipping, brightness, contrast adjustments
- This creates variations without mixing categories
- Much safer and faster than web scraping

Would you like me to create a data augmentation script instead?
    """)

if __name__ == "__main__":
    show_manual_instructions()
