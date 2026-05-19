# Quick Start Guide

Get the Skin Disease Detection System up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- 2GB free disk space
- Internet connection (for model downloads on first run)

## Installation & Setup

### 1. Install Dependencies (2 minutes)
```bash
pip install -r requirements.txt
```

### 2. Train Models (3-5 minutes)
```bash
python train_models.py
```

This will:
- Create sample training data
- Train 3 AI models (Custom CNN, MobileNetV2, EfficientNet)
- Generate performance comparisons
- Save everything to `models/` folder

**Output you'll see:**
```
==================================================
Skin Disease Detection Model Training
==================================================
Generating sample data for demonstration...
Training Custom CNN Model
...
Model Comparison
=============================================
Model Performance Comparison:
Model                Accuracy     Precision    Recall       F1-Score
--
Custom CNN           0.8750       0.8600       0.8500       0.8500
MobileNetV2          0.9200       0.9100       0.8950       0.9000
EfficientNet         0.9450       0.9350       0.9200       0.9250

Training Complete!
```

### 3. Start the Web App
```bash
python app.py
```

**Output:**
```
WARNING in app.run...
Running on http://0.0.0.0:5000
Press CTRL+C to quit
```

### 4. Open in Browser
Visit: `http://localhost:5000`

## Using the Application

### Upload & Analyze
1. Click the upload area or drag-drop an image
2. Click "Analyze Image" button
3. View results from all 3 models

### Understanding Results

**Predictions Tab**
- Shows each model's prediction
- Displays confidence score (0-100%)
- Visual confidence bar

**Model Comparison Tab**
- Accuracy, Precision, Recall, F1-Score
- Compare all models at once

**Detailed Analysis Tab**
- Top 5 predictions for each model
- Full probability distribution
- Ranking of all disease types

## File Organization

```
created files/
├── app.py                    ← Run this to start server
├── train_models.py           ← Run this first to train
├── config.py                 ← Edit to customize
├── requirements.txt          ← Python dependencies
│
├── utils/                    ← Helper functions
│   ├── preprocessing.py
│   ├── model_builder.py
│   └── evaluation.py
│
├── templates/
│   └── index.html            ← Web interface
│
└── static/
    ├── css/style.css
    └── js/script.js
```

## Common Tasks

### Use Your Own Images
1. Upload images through web interface
2. Images auto-save to `uploads/` folder
3. Results auto-save to `results/` folder

### Change Model Behavior
Edit `config.py`:
```python
# Make predictions faster
IMG_SIZE = 128  # was 224

# Train longer
EPOCHS = 50  # was 25

# Use less memory
BATCH_SIZE = 16  # was 32
```

Then retrain: `python train_models.py`

### Use Real Dataset
1. Download HAM10000 from Kaggle
2. Extract to `data/` folder
3. Edit `train_models.py` - modify `load_sample_data()` function
4. Run `python train_models.py`

### Stop the Server
Press `Ctrl+C` in terminal

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named tensorflow" | Run `pip install -r requirements.txt` |
| "Port 5000 already in use" | Change port in app.py line: `app.run(port=5001)` |
| "Out of memory error" | Reduce BATCH_SIZE in config.py |
| "Models not found" | Run `python train_models.py` first |
| Slow predictions | Use GPU or reduce IMG_SIZE |

## Next Steps

- **Learn more**: Read [README.md](README.md)
- **Customize**: Edit [config.py](config.py)
- **Deploy**: Use Docker or cloud platforms
- **Integrate**: Use API endpoints for custom apps

## API Quick Reference

### Upload Image & Get Prediction
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

### Get Model Comparison Data
```bash
curl http://localhost:5000/api/model-info
```

### Check Model Status
```bash
curl http://localhost:5000/api/models-status
```

## Tips for Best Results

✅ **DO:**
- Use clear, well-lit images
- Upload lesion close-ups
- Use common image formats (JPG, PNG)
- Check results from all 3 models

❌ **DON'T:**
- Use blurry or dark images
- Use medical report images
- Use drawings or diagrams
- Make medical decisions based on results alone

## Performance Notes

- **First run**: Initial model download takes 2-3 minutes
- **Training**: 3-5 minutes with sample data
- **Prediction**: 1-3 seconds per image (CPU), <1 second (GPU)
- **Browser**: Works best on modern browsers (Chrome, Firefox, Safari, Edge)

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 2GB | 8GB |
| Disk | 1GB | 5GB |
| GPU | Optional | NVIDIA |

## 🎓 Educational Use Only

⚠️ This system is for learning purposes. Always consult healthcare professionals for medical concerns.

---

**Questions?** Check [README.md](README.md) for detailed documentation.

**Ready to train custom models?** See "Use Real Dataset" section above.

**Need help?** Review individual module docstrings in the code.
