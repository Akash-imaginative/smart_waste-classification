import argparse
import os
from pathlib import Path
from typing import Dict
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU Detected: {len(gpus)} device(s)")
        print(f"  GPU Name: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("⚠ No GPU detected - running on CPU")

CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]

IMAGE_SIZE = (224, 224)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_and_preprocess(image_path: Path) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    arr = arr.astype("float32") / 255.0
    return arr  # Return without expand_dims for batch processing


def load_and_preprocess_batch(image_paths: list) -> np.ndarray:
    """Load and preprocess multiple images at once for faster GPU processing"""
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            img = load_and_preprocess(path)
            images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return np.array(images) if images else None, valid_paths


def evaluate(model_path: Path, data_dir: Path, batch_size: int = 32) -> Dict:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    print(f"\nLoading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully!\n")

    total_samples = 0
    total_correct = 0
    per_class_stats = {name: {"correct": 0, "total": 0} for name in CLASS_NAMES}
    
    # For confusion matrix
    y_true = []
    y_pred = []

    print(f"Evaluating model on dataset (Batch size: {batch_size})...")
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"Warning: Directory not found for class '{class_name}'")
            continue

        files = [class_dir / f for f in os.listdir(class_dir) 
                if Path(f).suffix.lower() in IMAGE_EXTENSIONS]
        print(f"Processing {class_name}: {len(files)} images", end='', flush=True)

        # Process in batches for speed
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            
            try:
                batch_images, valid_paths = load_and_preprocess_batch(batch_files)
                if batch_images is None or len(batch_images) == 0:
                    continue
                
                # Batch prediction - much faster!
                preds = model.predict(batch_images, verbose=0)
                pred_indices = np.argmax(preds, axis=1)
                
                for pred_index in pred_indices:
                    pred_label = CLASS_NAMES[pred_index]
                    
                    per_class_stats[class_name]["total"] += 1
                    total_samples += 1
                    
                    y_true.append(idx)
                    y_pred.append(pred_index)

                    if pred_label == class_name:
                        per_class_stats[class_name]["correct"] += 1
                        total_correct += 1
                        
            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue
        
        print(" ✓")

    overall_accuracy = (total_correct / total_samples) * 100 if total_samples else 0.0

    per_class_accuracy = {
        name: (stats["correct"] / stats["total"] * 100 if stats["total"] else 0.0)
        for name, stats in per_class_stats.items()
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        "overall_accuracy": overall_accuracy,
        "total_samples": total_samples,
        "total_correct": total_correct,
        "per_class_accuracy": per_class_accuracy,
        "per_class_stats": per_class_stats,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Waste Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate waste classification model accuracy")
    parser.add_argument(
        "--model",
        default="../models/densenet121_waste_classifier.keras",
        type=str,
        help="Path to the trained Keras model",
    )
    parser.add_argument(
        "--data",
        default="../data/garbage_classification",
        type=str,
        help="Path to the dataset directory with class subfolders",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_dir = Path(args.data)

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    metrics = evaluate(model_path, data_dir)

    print("\n" + "="*60)
    print("       WASTE CLASSIFICATION MODEL ACCURACY REPORT")
    print("="*60)
    print(f"\nDataset Information:")
    print(f"  Total Images Evaluated : {metrics['total_samples']}")
    print(f"  Correctly Classified   : {metrics['total_correct']}")
    print(f"  Incorrectly Classified : {metrics['total_samples'] - metrics['total_correct']}")
    
    print(f"\n{'='*60}")
    print(f"  OVERALL ACCURACY: {metrics['overall_accuracy']:.2f}%")
    print(f"{'='*60}")
    
    print(f"\nPer-Class Performance:")
    print(f"{'Class':<15} {'Accuracy':<12} {'Correct':<10} {'Total'}")
    print("-" * 60)
    
    for class_name in CLASS_NAMES:
        acc = metrics['per_class_accuracy'][class_name]
        stats = metrics['per_class_stats'][class_name]
        print(f"{class_name:<15} {acc:>6.2f}%       {stats['correct']:>4}/{stats['total']:<4}")
    
    # Calculate average precision, recall, f1
    if len(metrics['y_true']) > 0:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            metrics['y_true'], metrics['y_pred'], average='weighted', zero_division=0
        )
        
        print(f"\n{'='*60}")
        print(f"Additional Metrics (Weighted Average):")
        print(f"  Precision : {precision*100:.2f}%")
        print(f"  Recall    : {recall*100:.2f}%")
        print(f"  F1-Score  : {f1*100:.2f}%")
        print(f"{'='*60}\n")
    
    # Plot confusion matrix
    try:
        plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, 
                            save_path="backend/confusion_matrix.png")
    except Exception as e:
        print(f"Could not generate confusion matrix plot: {e}")
    
    # Model Assessment
    print("\n" + "="*60)
    print("              MODEL ASSESSMENT")
    print("="*60)
    
    accuracy = metrics['overall_accuracy']
    if accuracy >= 95:
        grade = "EXCELLENT ⭐⭐⭐⭐⭐"
        assessment = "Outstanding performance! Production-ready model."
    elif accuracy >= 90:
        grade = "VERY GOOD ⭐⭐⭐⭐"
        assessment = "Strong performance. Minor improvements possible."
    elif accuracy >= 85:
        grade = "GOOD ⭐⭐⭐"
        assessment = "Decent performance. Consider fine-tuning."
    elif accuracy >= 75:
        grade = "FAIR ⭐⭐"
        assessment = "Moderate performance. Needs improvement."
    else:
        grade = "NEEDS IMPROVEMENT ⭐"
        assessment = "Low performance. Consider model architecture changes."
    
    print(f"\nModel Grade: {grade}")
    print(f"Assessment: {assessment}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
