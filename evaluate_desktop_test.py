"""
Evaluate highaccuracy_86.63_epoch29.keras model on EXTERNAL test dataset.
This is the independent test set to verify TRUE model accuracy.

Test directory: C:/Users/Lenovo/OneDrive/Desktop/test
Model: models/checkpoints/highaccuracy_86.63_epoch29.keras
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

TEST_DIR = r'C:\Users\Lenovo\OneDrive\Desktop\test'
MODEL_PATH = os.path.join('models', 'checkpoints', 'highaccuracy_86.63_epoch29.keras')
OUTPUT_DIR = 'evaluation_desktop_test'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

print("="*80)
print("DENSENET121 MODEL EVALUATION - INDEPENDENT TEST SET")
print("="*80)
print(f"Model: {MODEL_PATH}")
print(f"Test Directory: {TEST_DIR}")
print("="*80)

# Data generator
print("\nLoading test data...")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

print(f"Found {test_generator.samples} images in {len(test_generator.class_indices)} classes")

# Load model
print("\nLoading DenseNet121 model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Predict
print("\nRunning predictions on independent test set...")
pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

# Overall accuracy
accuracy = np.mean(y_pred == y_true)
print(f"\n{'='*80}")
print(f"OVERALL ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*80}")

# Classification report
print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print(report)

# Per-class accuracy
print("\n" + "="*80)
print("PER-CLASS ACCURACY")
print("="*80)
per_class_accuracy = {}
for idx, class_name in enumerate(CLASS_NAMES):
    mask = (y_true == idx)
    correct = np.sum(y_pred[mask] == idx)
    total = np.sum(mask)
    acc = correct / total if total > 0 else 0
    per_class_accuracy[class_name] = acc
    print(f"{class_name:15s}: {acc*100:.2f}% ({correct}/{total} correct)")

# Confusion matrix with DenseNet121 branding
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Number of Predictions'},
            linewidths=0.5, linecolor='gray')

plt.title(f'Confusion Matrix - DenseNet121 Model\n(Independent Test Set Accuracy: {accuracy*100:.2f}%)', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')

# Add subtitle with model info
plt.text(6, -1.2, 'Architecture: DenseNet121 (Transfer Learning) + Custom Classification Head',
         ha='center', fontsize=11, style='italic', color='#555')
plt.text(6, -0.6, f'External Test Dataset: {test_generator.samples} images across 12 waste categories',
         ha='center', fontsize=10, color='#666')

plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, 'DenseNet121_Confusion_Matrix_Desktop_Test.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved confusion matrix to {cm_path}")

# Save detailed report
report_path = os.path.join(OUTPUT_DIR, 'classification_report_desktop_test.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DENSENET121 MODEL - INDEPENDENT TEST SET EVALUATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Directory: {TEST_DIR}\n")
    f.write(f"Total Images: {test_generator.samples}\n\n")
    f.write(f"OVERALL ACCURACY: {accuracy*100:.2f}%\n")
    f.write("="*80 + "\n\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write("-"*80 + "\n")
    f.write(report)
    f.write("\n\n" + "="*80 + "\n")
    f.write("PER-CLASS ACCURACY:\n")
    f.write("="*80 + "\n")
    for cls, acc in per_class_accuracy.items():
        f.write(f"{cls:15s}: {acc*100:.2f}%\n")
print(f"✅ Saved classification report to {report_path}")

# Save per-class accuracy JSON
acc_json_path = os.path.join(OUTPUT_DIR, 'per_class_accuracy_desktop_test.json')
with open(acc_json_path, 'w', encoding='utf-8') as jf:
    json.dump({
        'overall_accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'total_images': int(test_generator.samples),
        'test_directory': TEST_DIR
    }, jf, indent=2)
print(f"✅ Saved per-class accuracy JSON to {acc_json_path}")

print("\n" + "="*80)
print("EVALUATION COMPLETE - TRUE MODEL PERFORMANCE VERIFIED")
print("="*80)
print(f"\nThis is the REAL accuracy on unseen data: {accuracy*100:.2f}%")
print("This dataset is independent and provides accurate model assessment.")
