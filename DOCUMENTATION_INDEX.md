# Documentation Index

Complete guide to all documentation files in this project.

## Quick Start (Start Here!)

### For Impatient Users (5 minutes)
**File**: `QUICKSTART_HAM10000.md`
- Install dependencies
- Download dataset (one command)
- Train models (one command)
- Run web app (one command)

### For Detail-Oriented Users (30 minutes)
**File**: `TRAINING_WITH_HAM10000.md`
- Step-by-step Kaggle setup
- Detailed download instructions
- Complete training walkthrough
- How to monitor training
- How to test your models
- Troubleshooting guide

### For Command-Line Users (15 minutes)
**File**: `HAM10000_SETUP.md`
- Kaggle CLI setup
- Dataset directory structure
- Disease class mapping
- Training command options
- Troubleshooting with exact commands

## Main Documentation

### Project Overview
**File**: `README.md`
- What this project does
- Features and capabilities
- Installation instructions
- Configuration options
- API endpoints
- Usage examples
- Model descriptions

## How to Choose

### "I just want to run it"
→ `QUICKSTART_HAM10000.md`
- 4 simple shell commands
- Everything automated
- 30-60 min training time

### "I want to understand the process"
→ `TRAINING_WITH_HAM10000.md`
- Detailed explanations at each step
- Includes what's happening behind the scenes
- Includes monitoring and debugging
- Includes examples of testing

### "I need help with Kaggle/dataset"
→ `HAM10000_SETUP.md`
- Kaggle account setup
- API key configuration
- Manual command examples
- Verification steps
- Troubleshooting specifics

### "I want complete project documentation"
→ `README.md`
- Architecture overview
- All configuration options
- API documentation
- Model architecture details
- File structure

## File Structure

```
skin-disease-detection/
├── README.md                           (Main documentation)
├── QUICKSTART_HAM10000.md             (Fastest way to get started)
├── TRAINING_WITH_HAM10000.md          (Complete step-by-step guide)
├── HAM10000_SETUP.md                  (Dataset setup details)
├── DOCUMENTATION_INDEX.md             (This file)
│
├── app.py                              (Flask web app)
├── train_models.py                     (Train with sample data)
├── train_models_ham10000.py           (Train with real data - USE THIS!)
├── config.py                           (Configuration settings)
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py                (Image loading & preprocessing)
│   ├── model_builder.py               (Model architecture definitions)
│   ├── evaluation.py                  (Metrics & visualizations)
│   └── ham10000_loader.py            (HAM10000 data loading)
│
├── templates/
│   └── index.html                      (Web interface)
│
├── static/
│   ├── css/
│   │   └── style.css                  (Styling)
│   └── js/
│       └── script.js                  (JavaScript functionality)
│
├── models/                             (Saved trained models)
│   ├── custom_cnn_model.h5
│   ├── mobilenetv2_model.h5
│   └── efficientnet_model.h5
│
├── data/                               (Dataset folder)
│   └── HAM10000_images/               (Place dataset here)
│
├── results/                            (Training visualizations)
│   ├── *_training_history.png
│   ├── *_confusion_matrix.png
│   └── model_comparison.json
│
└── requirements.txt                    (Python dependencies)
```

## Training Workflows

### Standard Workflow (HAM10000)
```
1. Read: QUICKSTART_HAM10000.md
2. Install: pip install -r requirements.txt
3. Download: kaggle datasets download ...
4. Train: python train_models_ham10000.py
5. Run: python app.py
6. Open: http://localhost:5000
```

### Troubleshooting Workflow
```
1. Problem? → TRAINING_WITH_HAM10000.md → Troubleshooting section
2. Dataset issue? → HAM10000_SETUP.md → Verify dataset section
3. Configuration? → README.md → Configuration section
4. API question? → README.md → API Endpoints section
```

### Custom Setup Workflow
```
1. Read: README.md (Installation section)
2. Read: config.py (Understand configuration)
3. Read: HAM10000_SETUP.md (Dataset setup)
4. Customize: config.py (Your settings)
5. Run: train_models_ham10000.py
```

## Key Concepts

### Disease Classes (7 types)
- **Actinic keratosis** (AKIEC) - Precancerous lesions
- **Basal cell carcinoma** (BCC) - Most common skin cancer
- **Benign keratosis** (BKL) - Non-cancerous growths
- **Dermatofibroma** (DF) - Benign fibrous growth
- **Melanoma** (MEL) - Most serious skin cancer
- **Melanocytic nevus** (NV) - Common moles (67% of dataset)
- **Vascular lesion** (VASC) - Blood vessel lesions

### Models Trained
1. **Custom CNN** - Built from scratch
   - 4 convolutional blocks
   - Batch normalization
   - Dropout regularization
   - ~2M parameters

2. **MobileNetV2** - Transfer learning
   - Pre-trained on ImageNet
   - Lightweight and fast
   - ~3.5M parameters
   - Good for mobile deployment

3. **EfficientNet** - Transfer learning
   - Pre-trained on ImageNet
   - Balanced efficiency/accuracy
   - ~7M parameters
   - Best overall performance (usually)

*Note: ResNet50 excluded as requested*

### Dataset: HAM10000
- **Total images**: 10,015
- **Image size**: Variable (preprocessed to 224×224)
- **Classes**: 7 disease types
- **Train/Val split**: 80/20 (stratified)
- **Class imbalance**: Yes (handled with class weights)
- **Source**: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

## Common Tasks

### How to train with HAM10000?
→ `QUICKSTART_HAM10000.md` (Step 3: Train Models)

### How to download the dataset?
→ `QUICKSTART_HAM10000.md` (Step 1: Download HAM10000)

### How to run the web app?
→ `QUICKSTART_HAM10000.md` (Step 4: Test Web Interface)

### How to use predictions in my code?
→ `README.md` → API Endpoints section

### How to change BATCH_SIZE or EPOCHS?
→ `README.md` → Configuration section

### What if training runs out of memory?
→ `TRAINING_WITH_HAM10000.md` → Troubleshooting → Issue 3

### What if Kaggle download fails?
→ `HAM10000_SETUP.md` → Troubleshooting → Kaggle CLI Issues

### How to enable GPU training?
→ `TRAINING_WITH_HAM10000.md` → Prepare Your System → Optional GPU

### How to train with custom data?
→ `README.md` → create custom loader based on `ham10000_loader.py`

## Performance Expectations

### Training Time
- CPU: 1-1.5 hours
- GPU (RTX 2060): 10-15 minutes
- GPU (RTX 3080): 5-10 minutes

### Model Accuracy (with HAM10000)
- Custom CNN: 92-94%
- MobileNetV2: 94-96%
- EfficientNet: 95-97%

### Prediction Time
- Per image: 50-200ms (depending on model and hardware)

## Getting Help

### Error Messages?
1. Check `TRAINING_WITH_HAM10000.md` → Troubleshooting
2. Check `HAM10000_SETUP.md` → Troubleshooting
3. Check console output for exact error

### Conceptual Questions?
1. Check `README.md` for general info
2. Check `TRAINING_WITH_HAM10000.md` for step-by-step
3. Check code comments in Python files

### Dataset Questions?
1. Check `HAM10000_SETUP.md` → Dataset Overview
2. Check `TRAINING_WITH_HAM10000.md` → Train Models section
3. Check Kaggle dataset page: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

### Configuration Questions?
1. Check `README.md` → Configuration
2. Check `config.py` for comments
3. Check `TRAINING_WITH_HAM10000.md` → Troubleshooting

## Summary

**Start with**: `QUICKSTART_HAM10000.md` → **Get help with**: `TRAINING_WITH_HAM10000.md` → **Deep dive**: `README.md`

Good luck! 🎉
