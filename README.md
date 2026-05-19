# AI-Based Skin Disease Detection System

A production-ready web application for detecting skin diseases using deep learning models. This system uses multiple neural network architectures (Custom CNN, MobileNetV2, and EfficientNet) to classify skin lesions into seven disease categories.

## Features

- **Multiple AI Models**: Compares three deep learning architectures:
  - Custom CNN (built from scratch)
  - MobileNetV2 (transfer learning)
  - EfficientNet (transfer learning)

- **Comprehensive Analysis**:
  - Skin lesion image classification
  - Confidence scores for predictions
  - Probability distribution across disease classes
  - Model performance comparison

- **Disease Classification**:
  - Actinic keratosis
  - Basal cell carcinoma
  - Benign keratosis
  - Dermatofibroma
  - Melanoma
  - Melanocytic nevus
  - Vascular lesion

- **User-Friendly Interface**:
  - Clean, responsive web UI
  - Drag-and-drop image upload
  - Real-time image preview
  - Tabbed results display
  - Model comparison metrics

## System Requirements

- Python 3.8 or higher
- 4GB RAM (minimum), 8GB recommended
- GPU support recommended (NVIDIA CUDA for faster training)

## Quick Start

For the fastest setup using the HAM10000 dataset:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download HAM10000 dataset (see HAM10000_SETUP.md)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip -q skin-cancer-mnist-ham10000.zip -d data/
cd data/ && unzip -q HAM10000_images_part_1.zip && unzip -q HAM10000_images_part_2.zip && cd ..

# 3. Train models
python train_models_ham10000.py

# 4. Run web app
python app.py

# 5. Open http://localhost:5000
```

**See `QUICKSTART_HAM10000.md` for detailed instructions.**

## Installation

### 1. Clone or Download the Project
```bash
git clone <repository-url>
cd skin-disease-detection
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models

#### Option A: Train with HAM10000 Dataset (Recommended)
For production-ready models with real dermatoscopic images:

```bash
# 1. Set up Kaggle CLI and download dataset
#    (See HAM10000_SETUP.md for detailed instructions)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# 2. Extract dataset
unzip -q skin-cancer-mnist-ham10000.zip -d data/
cd data/
unzip -q HAM10000_images_part_1.zip
unzip -q HAM10000_images_part_2.zip
cd ..

# 3. Train models with real data
python train_models_ham10000.py
```

**Training time**: 30-60 minutes (depending on GPU availability)

#### Option B: Train with Sample Data (Fast Demonstration)
For quick testing without downloading the full dataset:

```bash
python train_models.py
```

**Training time**: 5-10 minutes

## Project Structure

```
skin-disease-detection/
├── app.py                  # Flask web application
├── config.py              # Configuration settings
├── train_models.py        # Model training script
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── utils/
│   ├── preprocessing.py   # Image preprocessing utilities
│   ├── model_builder.py   # Model architecture definitions
│   └── evaluation.py      # Model evaluation & visualization
│
├── templates/
│   └── index.html         # Main web interface
│
├── static/
│   ├── css/
│   │   └── style.css      # Stylesheet
│   └── js/
│       └── script.js      # Frontend JavaScript
│
├── models/                # Saved trained models (generated)
├── data/                  # Training data (user-provided)
├── uploads/               # Uploaded images for prediction
└── results/               # Generated visualizations & reports
```

## Usage

### Step 1: Train the Models

Before running the web application, you need to train the models. This will create the model files that the application needs.

```bash
python train_models.py
```

This script will:
- Generate sample training data
- Train the Custom CNN model
- Train the MobileNetV2 model
- Train the EfficientNet model
- Compare model performance
- Generate visualizations and metrics

**Note**: If you have your own dataset:
1. Place images in `data/` folder organized by disease class
2. Modify `load_sample_data()` in `train_models.py` to use your data
3. Run the training script

Expected output:
- Trained models saved in `models/`
- Performance visualizations in `results/`
- Model comparison metrics in `models/model_comparison.json`

### Step 2: Run the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Step 3: Use the Web Interface

1. Open `http://localhost:5000` in your browser
2. Upload a skin lesion image (PNG, JPG, JPEG, GIF, or BMP)
3. Click "Analyze Image" button
4. View predictions from all three models
5. Check model comparison metrics
6. Review detailed probability distributions

## Datasets

### HAM10000 Dataset (Recommended)

**10,015 real dermatoscopic images** from Kaggle:
- **Actinic keratosis**: 327 images (3.3%)
- **Basal cell carcinoma**: 376 images (3.8%)
- **Benign keratosis**: 1,099 images (11.0%)
- **Dermatofibroma**: 115 images (1.1%)
- **Melanoma**: 1,113 images (11.1%)
- **Melanocytic nevus**: 6,705 images (67.0%)
- **Vascular lesion**: 142 images (1.4%)

**Setup**: See `HAM10000_SETUP.md` or `QUICKSTART_HAM10000.md`

Features: Multi-source images, metadata, automatic class balancing

### Sample Data (for testing)

Quick test without downloading:
```bash
python train_models.py
```

Generates synthetic data for architecture validation.

## Configuration

Edit `config.py` to customize:

```python
# Image preprocessing
IMG_SIZE = 224              # Image resolution
BATCH_SIZE = 32             # Training batch size
VALIDATION_SPLIT = 0.2      # Validation data percentage

# Model training
EPOCHS = 25                 # Number of training epochs
LEARNING_RATE = 0.001       # Optimizer learning rate
RANDOM_SEED = 42            # Reproducibility seed

# File paths
MODELS_DIR = 'models'       # Saved models location
DATA_DIR = 'data'           # Training data location
UPLOAD_DIR = 'uploads'      # Uploaded images location
RESULTS_DIR = 'results'     # Results & visualizations location
```

## API Endpoints

### POST `/api/predict`
Upload an image and get predictions from all models.

**Request**:
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

**Response**:
```json
{
  "predictions": {
    "custom_cnn": {
      "class_index": 4,
      "class_name": "Melanoma",
      "confidence": 0.95,
      "all_probabilities": {
        "Actinic keratosis": 0.02,
        "Basal cell carcinoma": 0.01,
        ...
      }
    },
    "mobilenetv2": {...},
    "efficientnet": {...}
  },
  "best_model": "efficientnet",
  "image_filename": "image.jpg",
  "success": true
}
```

### GET `/api/model-info`
Get model comparison information and disease classes.

**Response**:
```json
{
  "model_info": {
    "custom_cnn": {
      "accuracy": 0.87,
      "precision": 0.86,
      "recall": 0.85,
      "f1_score": 0.85
    },
    ...
  },
  "available_models": ["custom_cnn", "mobilenetv2", "efficientnet"],
  "disease_classes": {...}
}
```

### GET `/api/models-status`
Check which models are currently loaded.

**Response**:
```json
{
  "loaded_models": ["custom_cnn", "mobilenetv2", "efficientnet"],
  "total_models": 3,
  "status": "ready"
}
```

## Model Architectures

### Custom CNN
- 4 convolutional blocks with batch normalization
- MaxPooling and dropout for regularization
- Fully connected layers with 512 and 256 neurons
- Parameter count: ~2.5M

### MobileNetV2
- Pre-trained on ImageNet
- Fine-tuned for skin disease classification
- Global average pooling + custom dense layers
- Optimized for mobile deployment (~3.5MB)

### EfficientNet
- EfficientNetB0 architecture
- Pre-trained on ImageNet
- Balanced efficiency and accuracy
- Parameter count: ~4.1M

## Performance Metrics

The system evaluates models using:

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / All positive predictions
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed misclassification patterns
- **ROC-AUC Curve**: Model discrimination ability

## Image Preprocessing

All images are preprocessed as follows:

1. **Loading**: Convert to RGB format
2. **Resizing**: 224×224 pixels
3. **Normalization**: Pixel values scaled to [0, 1]
4. **Data Augmentation** (training only):
   - Random rotation (±20°)
   - Width/height shift (±20%)
   - Random zoom (±20%)
   - Horizontal/vertical flips

## Troubleshooting

### Models not found error
```
Error: Models not loaded. Please train models first.
```
**Solution**: Run `python train_models.py` first

### Port already in use
```
Error: Address already in use
```
**Solution**: Change port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Out of memory error
**Solution**: Reduce `BATCH_SIZE` in `config.py`:
```python
BATCH_SIZE = 16  # or 8 for lower memory
```

### Slow predictions
**Solution**:
- Use GPU (install TensorFlow-GPU)
- Reduce image size: `IMG_SIZE = 128`
- Use fewer models for inference

## Dataset Information

The system is designed to work with the HAM10000 dataset, which contains:
- 10,015 dermatoscopic images
- 7 disease categories
- Images from ISIC and DermIS sources

To use HAM10000:
1. Download from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
2. Extract to `data/` folder
3. Organize by disease class
4. Modify `load_sample_data()` in `train_models.py`

## Deployment

### Local Deployment
```bash
python app.py
```

### Production Deployment (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t skin-disease-detector .
docker run -p 5000:5000 skin-disease-detector
```

## Important Disclaimers

⚠️ **IMPORTANT**: This system is for educational and research purposes only.

- **NOT a substitute for professional medical advice**
- **DO NOT use for self-diagnosis or treatment decisions**
- **Always consult with a dermatologist for medical concerns**
- **Model predictions should be verified by healthcare professionals**
- **Results may vary based on image quality and lighting conditions**

## Future Enhancements

- [ ] Real-time webcam input support
- [ ] Batch prediction for multiple images
- [ ] Advanced visualization (attention maps, CAM)
- [ ] Model explainability (SHAP, LIME)
- [ ] User authentication and prediction history
- [ ] Multi-language support
- [ ] Mobile app version
- [ ] Confidence threshold configuration
- [ ] Additional disease categories
- [ ] Continuous model retraining pipeline

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Citation

If you use this project in research, please cite:

```bibtex
@software{skin_disease_detection,
  author = {Your Name},
  title = {AI-Based Skin Disease Detection System},
  year = {2024},
  url = {https://github.com/username/repo}
}
```

## Support & Documentation

For issues, questions, or contributions:
- Create an issue on GitHub
- Check existing documentation
- Review API endpoint specifications
- Examine model architectures in `utils/model_builder.py`

## Technical Stack

- **Backend**: Python 3.8+, Flask
- **Deep Learning**: TensorFlow/Keras
- **Image Processing**: OpenCV, Pillow
- **Data Analysis**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **API**: RESTful Flask API with CORS support

## Acknowledgments

- HAM10000 dataset creators (ISIC Archive)
- TensorFlow and Keras teams
- Open-source community for amazing libraries

---

**Last Updated**: February 2024
**Version**: 1.0.0
**Status**: Production Ready (Educational Use)
