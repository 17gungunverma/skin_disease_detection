# HAM10000 Dataset Setup Guide

## Dataset Overview

HAM10000 is a large collection of multi-source dermatoscopic images of common pigmented skin lesions. It contains 10,015 images with 7 disease classes that match our system perfectly.

**Source:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

## Step-by-Step Setup

### 1. Create Kaggle Account & Get API Key

- Go to https://www.kaggle.com and create an account
- Click on your profile icon → Settings → API
- Click "Create New API Token"
- This downloads `kaggle.json` to your computer

### 2. Set Up Kaggle CLI

```bash
# Install kaggle CLI
pip install kaggle

# Create .kaggle directory
mkdir -p ~/.kaggle

# Move the kaggle.json file
# On Windows: Move it to C:\Users\<username>\.kaggle\
# On Mac/Linux: Move it to ~/.kaggle/
cp /path/to/kaggle.json ~/.kaggle/

# Set permissions (Mac/Linux only)
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download HAM10000 Dataset

```bash
# Download the dataset
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Unzip the dataset
unzip -q skin-cancer-mnist-ham10000.zip -d data/

# Verify the download
ls data/
# You should see: HAM10000_images_part_1.zip, HAM10000_images_part_2.zip, HAM10000_metadata.csv
```

### 4. Extract Images

```bash
# Extract both image parts
cd data/
unzip -q HAM10000_images_part_1.zip
unzip -q HAM10000_images_part_2.zip
cd ..

# Verify
ls data/HAM10000_images/
# You should see ~10,000 .jpg files
```

## Dataset Structure

After extraction, your data folder should look like:

```
data/
├── HAM10000_images/          (10,015 .jpg images)
├── HAM10000_metadata.csv     (image metadata and labels)
└── ISIC_metadata.csv         (additional metadata)
```

## Metadata Information

The HAM10000_metadata.csv contains:
- `image_id`: Unique image identifier
- `diagnosis`: Disease class (AKIEC, BCC, BKL, DF, MEL, NV, VASC)
- `lesion_id`: Lesion identifier
- `image_name`: Image filename
- `dx_type`: Diagnostic method
- `age`: Patient age
- `sex`: Patient sex
- `localization`: Body part location

## Disease Class Mapping

| Abbreviation | Full Name | Count |
|---|---|---|
| AKIEC | Actinic keratosis | ~327 |
| BCC | Basal cell carcinoma | ~376 |
| BKL | Benign keratosis | ~1099 |
| DF | Dermatofibroma | ~115 |
| MEL | Melanoma | ~1113 |
| NV | Melanocytic nevus | ~6705 |
| VASC | Vascular lesion | ~142 |

## Train with HAM10000

### Option 1: Use the provided training script

```bash
# Make sure dependencies are installed
pip install -r requirements.txt

# Run training with HAM10000
python train_models_ham10000.py
```

This will:
1. Load the metadata CSV
2. Split data into train/validation (80/20)
3. Preprocess images
4. Train Custom CNN, MobileNetV2, and EfficientNet
5. Generate evaluation metrics and comparison plots
6. Save trained models to `models/` folder

### Option 2: Manual training with custom parameters

```python
from utils.ham10000_loader import HAM10000DataLoader

# Load the dataset
loader = HAM10000DataLoader(data_dir='data/', metadata_file='data/HAM10000_metadata.csv')
(x_train, y_train), (x_val, y_val) = loader.load_data(test_size=0.2)

# Then use with your trainer
trainer = ModelTrainer()
trainer.train_custom_cnn(x_train, y_train, x_val, y_val)
# ... etc
```

## Training Tips

1. **First Run**: May take 30-60 minutes depending on hardware
2. **GPU Acceleration**: If you have a CUDA-capable GPU, it will be much faster
3. **Data Imbalance**: HAM10000 has class imbalance (NV: 6705 images vs DF: 115). The training script handles this with class weights
4. **Memory**: If you run out of memory, reduce BATCH_SIZE in config.py
5. **Progress**: You'll see validation metrics after each epoch

## Troubleshooting

### Kaggle CLI Issues
```bash
# Verify kaggle.json location
ls ~/.kaggle/kaggle.json

# Check permissions
chmod 600 ~/.kaggle/kaggle.json

# Test kaggle CLI
kaggle datasets list
```

### Image Loading Issues
- Ensure all 10,015 images are in `data/HAM10000_images/`
- Check file permissions: `chmod -R 755 data/HAM10000_images/`
- Verify metadata CSV format

### Out of Memory
- Reduce `BATCH_SIZE` in config.py (try 16 or 8)
- Reduce `IMG_SIZE` (currently 224, try 128)
- Train one model at a time instead of all three

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## After Training

Trained models will be saved to `models/`:
- `custom_cnn_model.h5` - Your custom CNN
- `mobilenetv2_model.h5` - MobileNetV2 transfer learning
- `efficientnet_model.h5` - EfficientNet transfer learning

Evaluation results will be in `results/`:
- Confusion matrices
- Training history plots
- Model comparison metrics

## Next Steps

Once trained with HAM10000:
1. Test predictions using the Flask web interface: `python app.py`
2. Upload real skin lesion images to get predictions
3. Compare model performance on the validation set

For more info on HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
