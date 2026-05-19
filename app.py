import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

from config import (
    MODELS_DIR, UPLOAD_DIR, DISEASE_CLASSES,
    CUSTOM_CNN_MODEL, MOBILENETV2_MODEL, EFFICIENTNET_MODEL
)
from utils.preprocessing import ImagePreprocessor
from utils.evaluation import ModelEvaluator

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Global models dictionary
models = {}
model_info = {}

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load trained models from disk."""
    global models, model_info
    
    print("[v0] Loading models...")
    
    try:
        # Load Custom CNN
        cnn_path = os.path.join(MODELS_DIR, CUSTOM_CNN_MODEL)
        if os.path.exists(cnn_path):
            models['custom_cnn'] = tf.keras.models.load_model(cnn_path)
            print(f"[v0] Loaded Custom CNN from {cnn_path}")
        
        # Load MobileNetV2
        mobile_path = os.path.join(MODELS_DIR, MOBILENETV2_MODEL)
        if os.path.exists(mobile_path):
            models['mobilenetv2'] = tf.keras.models.load_model(mobile_path)
            print(f"[v0] Loaded MobileNetV2 from {mobile_path}")
        
        # Load EfficientNet
        efficient_path = os.path.join(MODELS_DIR, EFFICIENTNET_MODEL)
        if os.path.exists(efficient_path):
            models['efficientnet'] = tf.keras.models.load_model(efficient_path)
            print(f"[v0] Loaded EfficientNet from {efficient_path}")
        
        # Load comparison data
        comparison_file = os.path.join(MODELS_DIR, 'model_comparison.json')
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r') as f:
                model_info = json.load(f)
                print(f"[v0] Loaded model comparison data")
        
        if not models:
            print("[v0] Warning: No models found. Please train models first using train_models.py")
        
    except Exception as e:
        print(f"[v0] Error loading models: {str(e)}")

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html', 
                         disease_classes=DISEASE_CLASSES,
                         available_models=list(models.keys()))

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
        
        # Check if models are loaded
        if not models:
            return jsonify({'error': 'Models not loaded. Please train models first.'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        preprocessor = ImagePreprocessor()
        pil_image = Image.open(filepath).convert('RGB')
        processed_image = preprocessor.load_and_preprocess_pil_image(pil_image)
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Get predictions from all models
        predictions = {}
        for model_name, model in models.items():
            try:
                pred = model.predict(processed_image, verbose=0)
                class_idx = np.argmax(pred[0])
                confidence = float(pred[0][class_idx])
                
                predictions[model_name] = {
                    'class_index': int(class_idx),
                    'class_name': DISEASE_CLASSES[class_idx],
                    'confidence': confidence,
                    'all_probabilities': {
                        DISEASE_CLASSES[i]: float(pred[0][i])
                        for i in range(len(DISEASE_CLASSES))
                    }
                }
            except Exception as e:
                print(f"[v0] Error in {model_name}: {str(e)}")
                predictions[model_name] = {'error': str(e)}
        
        # Determine best prediction (highest confidence)
        best_model = max(
            [(name, pred) for name, pred in predictions.items() 
             if 'error' not in pred],
            key=lambda x: x[1]['confidence'],
            default=(None, {})
        )
        
        response = {
            'predictions': predictions,
            'best_model': best_model[0],
            'image_filename': filename,
            'success': True
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[v0] Error in prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model comparison information."""
    try:
        return jsonify({
            'model_info': model_info,
            'available_models': list(models.keys()),
            'disease_classes': DISEASE_CLASSES
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models-status', methods=['GET'])
def models_status():
    """Get status of loaded models."""
    return jsonify({
        'loaded_models': list(models.keys()),
        'total_models': len(models),
        'status': 'ready' if models else 'no_models_loaded'
    }), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
