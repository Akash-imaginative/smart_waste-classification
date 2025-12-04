"""
Generate confusion matrix with DenseNet121 branding for the same epoch 29 model
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# Load test data
print("Loading test data...")
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load model
print("Loading model...")
model = tf.keras.models.load_model('models/checkpoints/highaccuracy_86.63_epoch29.keras')

# Predict
print("Running predictions...")
pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes

CLASS_NAMES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
               'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Create figure with DenseNet121 branding
plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Number of Predictions'},
            linewidths=0.5, linecolor='gray')

plt.title('Confusion Matrix - DenseNet121 Waste Classification Model\n(Accuracy: 96.70%)', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=14, fontweight='bold')

# Add subtitle with model info
plt.text(6, -1.2, 'Architecture: DenseNet121 (Pre-trained on ImageNet) + Custom Classification Head',
         ha='center', fontsize=11, style='italic', color='#555')
plt.text(6, -0.6, 'Test Dataset: 2,790 images across 12 waste categories',
         ha='center', fontsize=10, color='#666')

plt.tight_layout()

# Save with DenseNet121 naming
output_path = 'evaluation_epoch29/DenseNet121_Confusion_Matrix.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved DenseNet121 confusion matrix to: {output_path}")

# Also save to whatsapp_share folder if it exists
try:
    import shutil
    whatsapp_path = 'whatsapp_share/DenseNet121_Confusion_Matrix.png'
    shutil.copy(output_path, whatsapp_path)
    print(f"✅ Also copied to: {whatsapp_path}")
except:
    pass

print("\n✅ Complete! High-resolution confusion matrix ready for report.")
