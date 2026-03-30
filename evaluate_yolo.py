"""
YOLO Model Evaluation Script
=============================
This script evaluates the pre-trained YOLO11s.pt model and reports mAP metrics.
It does not modify any existing code files.
"""

import os
from ultralytics import YOLO
import time
from datetime import datetime

def create_dataset_yaml():
    """
    Create a dataset.yaml configuration file for YOLO validation.
    Adjust the paths according to your dataset structure.
    """
    yaml_content = """# Dataset configuration for YOLO validation
path: .  # Root directory (relative to this file)
train: data/train  # Training images directory
val: data/val      # Validation images directory
test: data/test    # Test images directory (optional)

# Number of classes
nc: 12

# Class names (waste categories)
names:
  0: battery
  1: biological
  2: brown-glass
  3: cardboard
  4: clothes
  5: green-glass
  6: metal
  7: paper
  8: plastic
  9: shoes
  10: trash
  11: white-glass
"""
    
    yaml_path = 'dataset.yaml'
    if not os.path.exists(yaml_path):
        print(f"📝 Creating {yaml_path}...")
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print(f"✅ Created {yaml_path}")
    else:
        print(f"✅ Found existing {yaml_path}")
    
    return yaml_path


def evaluate_yolo_model():
    """
    Load YOLO11s.pt model and evaluate it on the validation dataset.
    """
    print("\n" + "="*70)
    print("🚀 YOLO MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Create dataset configuration if it doesn't exist
    dataset_yaml = create_dataset_yaml()
    
    # Check if model file exists
    model_path = 'backend/yolo11s.pt'
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    print(f"📦 Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    print(f"✅ Model loaded successfully\n")
    
    # Create results directory
    results_dir = 'yolo_evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f'run_{timestamp}')
    
    print(f"📊 Running validation on dataset: {dataset_yaml}")
    print(f"💾 Results will be saved to: {run_dir}\n")
    
    # Start timing
    start_time = time.time()
    
    # Run validation
    # Parameters:
    # - data: path to dataset YAML
    # - conf: confidence threshold (0.25 is default)
    # - iou: IoU threshold for NMS (0.45 is default)
    # - project: directory to save results
    # - name: name of this run
    # - save_json: save results in COCO JSON format
    # - plots: save plots (confusion matrix, PR curves, etc.)
    try:
        metrics = model.val(
            data=dataset_yaml,
            conf=0.25,
            iou=0.45,
            project=results_dir,
            name=f'run_{timestamp}',
            save_json=True,
            plots=True,
            verbose=True
        )
        
        # End timing
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # Print results
        print("\n" + "="*70)
        print("📈 EVALUATION RESULTS")
        print("="*70)
        print(f"✅ Validation completed successfully!")
        print(f"⏱️  Evaluation time: {evaluation_time:.2f} seconds\n")
        
        print("📊 DETECTION METRICS:")
        print(f"   mAP@0.5:         {metrics.box.map50:.4f}  (IoU threshold = 0.5)")
        print(f"   mAP@0.5:0.95:    {metrics.box.map:.4f}  (COCO standard)")
        print(f"   Precision:       {metrics.box.mp:.4f}")
        print(f"   Recall:          {metrics.box.mr:.4f}")
        print(f"   F1 Score:        {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6):.4f}")
        
        print("\n📁 SAVED FILES:")
        print(f"   Results directory: {run_dir}")
        print(f"   - Confusion Matrix")
        print(f"   - Precision-Recall Curves")
        print(f"   - F1-Confidence Curve")
        print(f"   - Results in JSON format")
        print("="*70 + "\n")
        
        # Save summary to text file
        summary_path = os.path.join(run_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("YOLO MODEL EVALUATION SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Dataset: {dataset_yaml}\n")
            f.write(f"Evaluation Time: {evaluation_time:.2f} seconds\n\n")
            f.write("METRICS:\n")
            f.write(f"  mAP@0.5:      {metrics.box.map50:.4f}\n")
            f.write(f"  mAP@0.5:0.95: {metrics.box.map:.4f}\n")
            f.write(f"  Precision:    {metrics.box.mp:.4f}\n")
            f.write(f"  Recall:       {metrics.box.mr:.4f}\n")
        
        print(f"📄 Summary saved to: {summary_path}\n")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {str(e)}")
        print(f"   Make sure your validation dataset is properly formatted.")
        print(f"   YOLO expects images and corresponding .txt label files.")
        return
    
    print("✅ Evaluation complete!\n")


if __name__ == "__main__":
    evaluate_yolo_model()
