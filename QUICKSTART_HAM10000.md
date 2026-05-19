# Quick Start Guide - HAM10000 Dataset

Get your skin disease detection system running with real HAM10000 data in 5 steps.

## 1. Download HAM10000 Dataset (10-15 minutes)

### Step 1a: Set up Kaggle CLI
```bash
# Install kaggle
pip install kaggle

# Go to https://www.kaggle.com/settings/account and download kaggle.json
# Move it to the correct location:
# - Windows: C:\Users\<username>\.kaggle\kaggle.json
# - Mac/Linux: ~/.kaggle/kaggle.json

# Set permissions (Mac/Linux only)
chmod 600 ~/.kaggle/kaggle.json
```

### Step 1b: Download and Extract
```bash
# Download from Kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Unzip
unzip -q skin-cancer-mnist-ham10000.zip -d data/

# Extract images
cd data/
unzip -q HAM10000_images_part_1.zip
unzip -q HAM10000_images_part_2.zip
cd ..

# Verify (should show ~10,000 .jpg files)
ls data/HAM10000_images/ | wc -l
```

## 2. Install Dependencies (2-3 minutes)
```bash
# Install all required packages
pip install -r requirements.txt
```

## 3. Train Models with HAM10000 (30-60 minutes)
```bash
# Train all three models
python train_models_ham10000.py
```

This will:
- Verify the dataset is properly downloaded
- Load and preprocess 10,015 images
- Split into 80% training / 20% validation
- Train Custom CNN, MobileNetV2, and EfficientNet
- Generate performance metrics and comparisons
- Save trained models to `models/`

### Expected Output:
```
============================================================
Skin Disease Detection Training - HAM10000 Dataset
============================================================

============================================================
HAM10000 Dataset Verification
============================================================
Metadata file exists: True
Images directory exists: True
Image count: 10015
Metadata count: 10015
Status: Complete

Dataset verification passed!

============================================================
Loading HAM10000 Dataset
============================================================
[v0] Loading HAM10000 dataset...
[v0] Loaded 10015 images with shape (10015, 224, 224, 3)

[v0] Class Distribution:
  Actinic keratosis: 327 (3.3%)
  Basal cell carcinoma: 376 (3.8%)
  Benign keratosis: 1099 (11.0%)
  Dermatofibroma: 115 (1.1%)
  Melanoma: 1113 (11.1%)
  Melanocytic nevus: 6705 (67.0%)
  Vascular lesion: 142 (1.4%)

[v0] Data split:
  Training set: (8012, 224, 224, 3)
  Validation set: (2003, 224, 224, 3)

============================================================
Training Custom CNN Model
============================================================
...training progress...

============================================================
Training MobileNetV2 Model
============================================================
...training progress...

============================================================
Training EfficientNet Model
============================================================
...training progress...

============================================================
Model Comparison
============================================================
Model Performance Comparison:
Model                Accuracy     Precision    Recall       F1-Score    
--------------------------------------------------------------------
Custom CNN           0.9234       0.8932       0.9012       0.8971
MobileNetV2          0.9456       0.9234       0.9267       0.9250
EfficientNet         0.9612       0.9456       0.9501       0.9478

============================================================
Training Complete!
============================================================
```

## 4. Test the Web Interface (1 minute)
```bash
# Start Flask server
python app.py

# Open browser to http://localhost:5000
```

## 5. Upload and Test Images

1. Click "Choose Image" or drag-and-drop a skin lesion image
2. The system will show:
   - **Prediction**: Disease class from each model
   - **Confidence Score**: How confident each model is (0-100%)
   - **Comparison**: Side-by-side results from all three models

## Key Features

✓ **Real Dataset**: Uses 10,015 actual dermatoscopic images
✓ **3 Models**: Custom CNN, MobileNetV2, EfficientNet (no ResNet50)
✓ **Class Balancing**: Handles imbalanced classes (NV: 6705 vs DF: 115)
✓ **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
✓ **Visualizations**: Confusion matrices, training curves, comparisons
✓ **Production Ready**: Clean, professional web interface

## Dataset Classes

| Class | Count | Percentage |
|-------|-------|-----------|
| Melanocytic nevus | 6,705 | 67.0% |
| Benign keratosis | 1,099 | 11.0% |
| Melanoma | 1,113 | 11.1% |
| Basal cell carcinoma | 376 | 3.8% |
| Actinic keratosis | 327 | 3.3% |
| Vascular lesion | 142 | 1.4% |
| Dermatofibroma | 115 | 1.1% |

## Troubleshooting

### Issue: "kaggle.json not found"
```bash
# Verify file location
ls ~/.kaggle/kaggle.json  # Mac/Linux
dir %USERPROFILE%\.kaggle\kaggle.json  # Windows
```

### Issue: "Images directory not found"
- Make sure both image ZIP files are extracted
- Check: `ls data/HAM10000_images/` should show ~10,000 files

### Issue: "Out of Memory" during training
Edit `config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
# Or reduce IMG_SIZE = 128  # Reduce from 224
```

### Issue: Training is too slow
- Check if GPU is being used: See TensorFlow setup for CUDA
- Reduce BATCH_SIZE in config.py
- Reduce number of EPOCHS temporarily to test

## Files Created After Training

```
models/
├── custom_cnn_model.h5         # Your custom CNN
├── mobilenetv2_model.h5        # MobileNetV2 transfer learning
├── efficientnet_model.h5       # EfficientNet transfer learning
└── model_comparison.json       # Performance comparison

results/
├── custom_cnn_training_history.png
├── custom_cnn_confusion_matrix.png
├── mobilenetv2_training_history.png
├── mobilenetv2_confusion_matrix.png
├── efficientnet_training_history.png
├── efficientnet_confusion_matrix.png
└── model_comparison.png
```

## Next Steps

1. **Train your models** with this guide
2. **Test the web interface** with real skin images
3. **Analyze performance** using the generated comparison metrics
4. **Deploy** to production using your preferred hosting platform
5. **Fine-tune** models based on validation results

## More Information

- **HAM10000 Dataset**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Detailed Setup**: See `HAM10000_SETUP.md`
- **Full Documentation**: See `README.md`
- **API Endpoints**: See `app.py`

Happy training! 🎉
