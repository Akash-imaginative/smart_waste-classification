# GPU Setup Guide for TensorFlow (RTX 3050)

## Current Status
- **GPU**: NVIDIA RTX 3050 (6GB VRAM)
- **TensorFlow**: 2.20.0 (CPU-only version)
- **Status**: GPU not being utilized ❌

## Quick Fix: Install TensorFlow with GPU Support

### Option 1: Upgrade Current Installation (Recommended)

```powershell
# Uninstall CPU-only version
pip uninstall tensorflow

# Install GPU-enabled version
pip install tensorflow[and-cuda]
```

This will automatically install:
- TensorFlow with GPU support
- CUDA libraries
- cuDNN libraries

**No manual CUDA installation needed!** (TensorFlow 2.20+ bundles everything)

### Option 2: Manual CUDA Setup (Advanced)

If Option 1 doesn't work:

1. **Install CUDA Toolkit 12.6**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Select: Windows > x86_64 > 11 > exe (local)

2. **Install cuDNN 9.x**
   - Download: https://developer.nvidia.com/cudnn
   - Extract to CUDA directory

3. **Install GPU TensorFlow**
   ```powershell
   pip install tensorflow-gpu
   ```

## Verify GPU Installation

```powershell
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Performance Improvement with GPU

### Current (CPU):
- Speed: ~850-1,150 images/minute
- Total time: **10-12 minutes**

### With GPU (RTX 3050):
- Speed: ~8,000-12,000 images/minute
- Total time: **1-2 minutes** ⚡

**10x faster evaluation!**

## After GPU Setup

The optimized evaluation script will automatically:
- Detect and use your GPU
- Process images in batches (32 at a time)
- Show real-time progress
- Complete in ~1-2 minutes instead of 10-12 minutes

## Troubleshooting

### Issue: GPU not detected
```powershell
# Check NVIDIA driver
nvidia-smi
```

### Issue: CUDA version mismatch
```powershell
# Check CUDA version
nvcc --version
```

### Issue: Out of memory
Edit `evaluate_model.py` and reduce batch size:
```python
metrics = evaluate(model_path, data_dir, batch_size=16)  # Default is 32
```

## Quick Install Command

```powershell
# In your venv
cd C:\Users\Lenovo\OneDrive\Desktop\Waste-Classification
.\venv\Scripts\activate
pip uninstall tensorflow -y
pip install tensorflow[and-cuda]
```

Then run evaluation again:
```powershell
cd backend
python evaluate_model.py
```

---

**Note**: The current evaluation script is already optimized with batch processing. Once GPU is enabled, it will automatically use it with no code changes needed!
