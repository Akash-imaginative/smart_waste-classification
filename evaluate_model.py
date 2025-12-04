"""
Model Evaluation Script
Calculates accuracy, confusion matrix, precision, recall, and F1-score for the waste classification model
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MODEL_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\Waste-Classification\models\densenet121_waste_classifier.keras"
TEST_DATA_DIR = r"C:\Users\Lenovo\OneDrive\Desktop\test"  # Using organized data
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Class names (must match training order)
CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

def load_model():
    """Load the trained model"""
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully!")
    return model

def prepare_test_data():
    """Prepare test data generator"""
    print("\nPreparing test data...")
    
    # Create test data generator (no augmentation, only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False  # Important for evaluation!
    )
    
    print(f"✅ Found {test_generator.samples} test images")
    print(f"✅ Classes: {list(test_generator.class_indices.keys())}")
    
    return test_generator

def evaluate_model(model, test_generator):
    """Evaluate model and generate metrics"""
    print("\n" + "="*70)
    print("🔍 EVALUATING MODEL")
    print("="*70)
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Calculate overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Display overall accuracy prominently
    print("\n" + "="*70)
    print("🎯 OVERALL TEST ACCURACY")
    print("="*70)
    print(f"\n   Accuracy: {accuracy*100:.2f}%")
    print(f"   Correct predictions: {np.sum(y_true == y_pred)}/{len(y_true)}")
    print(f"   Incorrect predictions: {np.sum(y_true != y_pred)}/{len(y_true)}")
    
    # Generate classification report
    print("\n" + "="*70)
    print("📊 CLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)
    
    # Generate confusion matrix
    print("\n" + "="*70)
    print("📈 CONFUSION MATRIX")
    print("="*70)
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Waste Classification Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to: {cm_path}")
    plt.show()
    
    # Calculate per-class metrics
    print("\n" + "="*70)
    print("📋 PER-CLASS METRICS")
    print("="*70)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    print(f"\n{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:<20} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # Calculate macro and weighted averages
    print("-" * 70)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print(f"{'Macro Avg':<20} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}")
    print(f"{'Weighted Avg':<20} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    
    # Plot per-class metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x_pos = np.arange(len(CLASS_NAMES))
    
    # Precision
    axes[0].bar(x_pos, precision, color='skyblue', edgecolor='navy')
    axes[0].set_xlabel('Classes', fontweight='bold')
    axes[0].set_ylabel('Precision', fontweight='bold')
    axes[0].set_title('Precision by Class', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1].bar(x_pos, recall, color='lightcoral', edgecolor='darkred')
    axes[1].set_xlabel('Classes', fontweight='bold')
    axes[1].set_ylabel('Recall', fontweight='bold')
    axes[1].set_title('Recall by Class', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1-Score
    axes[2].bar(x_pos, f1, color='lightgreen', edgecolor='darkgreen')
    axes[2].set_xlabel('Classes', fontweight='bold')
    axes[2].set_ylabel('F1-Score', fontweight='bold')
    axes[2].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    metrics_path = 'per_class_metrics.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Per-class metrics chart saved to: {metrics_path}")
    plt.show()
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    print(f"Total test samples: {len(y_true)}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Macro-averaged F1-Score: {macro_f1:.4f}")
    print(f"Weighted-averaged F1-Score: {weighted_f1:.4f}")
    
    # Identify best and worst performing classes
    best_class_idx = np.argmax(f1)
    worst_class_idx = np.argmin(f1)
    
    print(f"\n🏆 Best performing class: {CLASS_NAMES[best_class_idx]} (F1: {f1[best_class_idx]:.4f})")
    print(f"⚠️  Worst performing class: {CLASS_NAMES[worst_class_idx]} (F1: {f1[worst_class_idx]:.4f})")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': support
    }

def main():
    """Main evaluation function"""
    print("="*70)
    print("🎯 WASTE CLASSIFICATION MODEL EVALUATION")
    print("="*70)
    
    # Check if test directory exists
    if not os.path.exists(TEST_DATA_DIR):
        print(f"❌ Error: Test directory not found: {TEST_DATA_DIR}")
        print("Please make sure you have a 'test' folder with images organized by class.")
        return
    
    # Load model
    model = load_model()
    
    # Prepare test data
    test_generator = prepare_test_data()
    
    # Evaluate model
    results = evaluate_model(model, test_generator)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  - confusion_matrix.png")
    print("  - per_class_metrics.png")
    print("\n💡 Use these metrics to understand your model's strengths and weaknesses!")

if __name__ == "__main__":
    main()
