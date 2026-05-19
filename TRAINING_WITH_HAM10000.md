# Complete Training Guide: HAM10000 Dataset

This guide walks you through training skin disease detection models using the HAM10000 dataset from Kaggle.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Download HAM10000](#download-ham10000)
3. [Prepare Your System](#prepare-your-system)
4. [Train Models](#train-models)
5. [Monitor Training](#monitor-training)
6. [Test Models](#test-models)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- Python 3.8+
- 4GB RAM minimum (8GB+ recommended)
- GPU optional but recommended (NVIDIA CUDA for 5-10x speedup)
- Kaggle account (free at https://www.kaggle.com)

---

## Download HAM10000

### Step 1: Get Kaggle API Key

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token"
3. File `kaggle.json` downloads automatically

### Step 2: Set Up Kaggle CLI

```bash
# Install kaggle package
pip install kaggle

# Create .kaggle directory (if it doesn't exist)
mkdir -p ~/.kaggle              # Mac/Linux
mkdir %USERPROFILE%\.kaggle     # Windows

# Move kaggle.json to the correct location
# Copy the downloaded kaggle.json to:
# - Mac/Linux: ~/.kaggle/kaggle.json
# - Windows: C:\Users\<username>\.kaggle\kaggle.json

# Set permissions (Mac/Linux only)
chmod 600 ~/.kaggle/kaggle.json

# Verify setup
kaggle datasets list  # Should show datasets from Kaggle
```

### Step 3: Download and Extract Dataset

```bash
# Download HAM10000
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Unzip main archive
unzip -q skin-cancer-mnist-ham10000.zip -d data/

# Extract image parts (this takes a few minutes)
cd data/
unzip -q HAM10000_images_part_1.zip
unzip -q HAM10000_images_part_2.zip
cd ..

# Verify download (should show ~10,015 files)
ls data/HAM10000_images/ | wc -l

# Verify structure
ls -la data/
# Should show:
# - HAM10000_metadata.csv
# - HAM10000_images/
# - (other supporting files)
```

**Expected output**:
```
data/
├── HAM10000_metadata.csv      (10,015 records)
├── HAM10000_images/            (~10,015 .jpg files)
├── HAM10000_images_part_1.zip
├── HAM10000_images_part_2.zip
└── ...
```

**Total download**: ~1.2 GB
**After extraction**: ~1.8 GB

---

## Prepare Your System

### Step 1: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow; print(f'TensorFlow version: {tensorflow.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Step 2: Verify Setup

```bash
# Python script to verify everything is ready
python -c "
from utils.ham10000_loader import HAM10000DataLoader
result = HAM10000DataLoader.verify_dataset(data_dir='data/')
print('Dataset Verification Results:')
for key, value in result.items():
    print(f'  {key}: {value}')
"
```

Expected output:
```
Dataset Verification Results:
  metadata_exists: True
  images_dir_exists: True
  image_count: 10015
  metadata_count: 10015
  status: Complete
```

### Step 3: (Optional) Enable GPU

For faster training with NVIDIA GPU:

```bash
# Check if GPU is available
python -c "import tensorflow as tf; print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"

# Install GPU support (CUDA)
# Follow: https://www.tensorflow.org/install/source#gpu
```

---

## Train Models

### Quick Start (Recommended)

```bash
# Train all three models with HAM10000
python train_models_ham10000.py
```

### What This Does

1. **Verifies** HAM10000 dataset is complete
2. **Loads** all 10,015 images and preprocesses them
3. **Splits** data into 80% training (8,012) / 20% validation (2,003)
4. **Trains** three models:
   - Custom CNN (built from scratch)
   - MobileNetV2 (transfer learning)
   - EfficientNet (transfer learning)
5. **Evaluates** each model with metrics and visualizations
6. **Compares** all three models
7. **Saves** trained models to `models/` folder

### Training Time Estimates

| Hardware | Custom CNN | MobileNetV2 | EfficientNet | Total |
|----------|-----------|------------|-------------|-------|
| CPU only | 15-20 min | 20-25 min | 30-40 min | 65-85 min |
| GPU (RTX 2060) | 2-3 min | 3-4 min | 4-5 min | 9-12 min |
| GPU (RTX 3080) | 1 min | 1-2 min | 2 min | 4-5 min |

---

## Monitor Training

### During Training

Watch the console output:

```
============================================================
Training Custom CNN Model
============================================================
Epoch 1/25
252/252 [==============================] - 45s 179ms/step - loss: 1.8234 - accuracy: 0.5123 - val_loss: 1.5678 - val_accuracy: 0.6234
Epoch 2/25
252/252 [==============================] - 42s 167ms/step - loss: 1.4567 - accuracy: 0.6456 - val_loss: 1.2345 - val_accuracy: 0.7123
...
```

**Key metrics to watch**:
- `loss`: Training loss (should decrease)
- `accuracy`: Training accuracy (should increase)
- `val_loss`: Validation loss
- `val_accuracy`: Validation accuracy (best model restored when improves)

### After Training

Results saved to:
- **Models**: `models/*.h5`
- **Visualizations**: `results/` folder

Example files created:
```
results/
├── custom_cnn_training_history.png
├── custom_cnn_confusion_matrix.png
├── mobilenetv2_training_history.png
├── mobilenetv2_confusion_matrix.png
├── efficientnet_training_history.png
├── efficientnet_confusion_matrix.png
├── model_comparison.png
└── (tensorboard logs)

models/
├── custom_cnn_model.h5
├── mobilenetv2_model.h5
├── efficientnet_model.h5
└── model_comparison.json
```

---

## Test Models

### Option 1: Web Interface (Recommended)

```bash
# Start Flask server
python app.py

# Open http://localhost:5000 in browser
```

Then:
1. Upload a skin lesion image (JPG, PNG, etc.)
2. Click "Analyze"
3. View predictions from all three models
4. Compare confidence scores and probabilities

### Option 2: Python API

```python
from tensorflow.keras.models import load_model
from PIL import Image
from utils.preprocessing import ImagePreprocessor

# Load a trained model
model = load_model('models/custom_cnn_model.h5')

# Load and preprocess image
preprocessor = ImagePreprocessor()
image = preprocessor.load_and_preprocess_image('path/to/image.jpg')
image = image.reshape(1, 224, 224, 3)

# Get prediction
prediction = model.predict(image)
class_index = prediction.argmax()
confidence = prediction[0, class_index]

# Disease classes
classes = {
    0: 'Actinic keratosis',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevus',
    6: 'Vascular lesion'
}

print(f"Predicted class: {classes[class_index]}")
print(f"Confidence: {confidence * 100:.2f}%")
```

### Option 3: API Endpoint

```bash
# Start Flask server
python app.py

# Make a prediction via API
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict

# Get model status
curl http://localhost:5000/api/models-status
```

---

## Troubleshooting

### Issue 1: "kaggle.json not found"

**Solution**:
```bash
# Verify file location
ls ~/.kaggle/kaggle.json              # Mac/Linux
dir %USERPROFILE%\.kaggle\kaggle.json # Windows

# If not found, check Kaggle account settings
# https://www.kaggle.com/settings/account
# Click "Create New API Token" again
```

### Issue 2: "Images directory not found"

**Solution**:
```bash
# Verify extraction
ls data/HAM10000_images/ | head -5
# Should show image filenames like:
# ISIC_0024306.jpg
# ISIC_0024307.jpg
# ...

# If missing, re-extract:
cd data/
unzip -q HAM10000_images_part_1.zip
unzip -q HAM10000_images_part_2.zip
cd ..
```

### Issue 3: "Out of Memory" during training

**Solutions** (in order of preference):

```python
# Edit config.py - Option 1: Reduce batch size
BATCH_SIZE = 16  # Was 32

# Option 2: Reduce image size
IMG_SIZE = 128  # Was 224

# Option 3: Train one model at a time
# Edit train_models_ham10000.py to comment out some trains
```

### Issue 4: Training is very slow

**Check GPU usage**:
```bash
# Monitor GPU during training
nvidia-smi -l 1  # Updates every 1 second (Ctrl+C to stop)

# If GPU not used, install TensorFlow GPU support
# https://www.tensorflow.org/install/gpu
```

**Other optimizations**:
- Reduce `EPOCHS` in config.py temporarily
- Reduce `BATCH_SIZE` (smaller batches = faster iterations)
- Close other applications using CPU/GPU

### Issue 5: "CUDA out of memory"

**Solutions**:
```python
# Edit config.py
BATCH_SIZE = 8   # Further reduce batch size

# Or clear TensorFlow cache
import tensorflow as tf
tf.keras.backend.clear_session()
```

### Issue 6: Models not improving (high loss)

**Possible causes**:
- Image preprocessing issue
- Class labels misaligned
- Learning rate too high/low
- Dataset corruption

**Verify dataset**:
```python
from utils.ham10000_loader import HAM10000DataLoader
loader = HAM10000DataLoader()
(x_train, y_train), (x_val, y_val) = loader.load_data()

print(f"Data shapes: {x_train.shape}, {y_train.shape}")
print(f"Data range: {x_train.min()} to {x_train.max()}")
print(f"Label distribution: {y_train.sum(axis=0)}")
```

---

## Next Steps

1. **Review Results**: Check `results/` folder for visualizations
2. **Compare Models**: See `model_comparison.json` for metrics
3. **Deploy**: Upload to cloud platform
4. **Fine-tune**: Experiment with hyperparameters
5. **Collect More Data**: Add custom images to training set

---

## FAQ

**Q: Can I stop training and resume?**
A: Not directly, but you can save/load model checkpoints. See Keras callbacks documentation.

**Q: How do I use my own images?**
A: Place images in a folder with the same structure as HAM10000_images, update the loader path.

**Q: Can I train with a smaller dataset?**
A: Yes, but with less data, accuracy will be lower. HAM10000 is optimized for this system.

**Q: Do I need GPU?**
A: No, but it's 5-10x faster. Training on CPU takes 1-1.5 hours.

**Q: How often should I retrain?**
A: Retrain when adding new data or if model performance degrades.

---

## Resources

- **HAM10000 Dataset**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- **Kaggle API**: https://github.com/Kaggle/kaggle-api
- **TensorFlow Docs**: https://www.tensorflow.org/guide
- **Paper**: https://arxiv.org/abs/1803.10417

---

For questions or issues, check the main README.md or HAM10000_SETUP.md files.

Good luck with your training! 🎉
