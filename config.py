import os

# Image preprocessing config
IMG_SIZE = 64
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2

# Model training config
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Skin disease classes
DISEASE_CLASSES = {
    0: 'Actinic keratosis',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevus',
    6: 'Vascular lesion'
}

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Ensure directories exist
for directory in [MODELS_DIR, DATA_DIR, UPLOAD_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model names
CUSTOM_CNN_MODEL = 'custom_cnn_model.h5'
MOBILENETV2_MODEL = 'mobilenetv2_model.h5'
EFFICIENTNET_MODEL = 'efficientnet_model.h5'
