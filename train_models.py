import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
import json

from config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS, DISEASE_CLASSES, 
    MODELS_DIR, DATA_DIR, CUSTOM_CNN_MODEL, 
    MOBILENETV2_MODEL, EFFICIENTNET_MODEL, RANDOM_SEED
)
from utils.preprocessing import ImagePreprocessor
from utils.model_builder import ModelBuilder
from utils.evaluation import ModelEvaluator

np.random.seed(RANDOM_SEED)

class ModelTrainer:
    """Trains skin disease detection models."""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.builder = ModelBuilder()
        self.evaluator = ModelEvaluator()
        self.models_info = {}
    
    def load_sample_data(self):
        """Load sample training data (for demonstration)."""
        print("[v0] Generating sample data for demonstration...")
        
        # Create synthetic data for training
        num_samples = 500
        x_train = np.random.rand(num_samples, IMG_SIZE, IMG_SIZE, 3).astype('float32')
        y_train = np.zeros((num_samples, len(DISEASE_CLASSES)))
        for i in range(num_samples):
            y_train[i, np.random.randint(0, len(DISEASE_CLASSES))] = 1
        
        x_val = np.random.rand(100, IMG_SIZE, IMG_SIZE, 3).astype('float32')
        y_val = np.zeros((100, len(DISEASE_CLASSES)))
        for i in range(100):
            y_val[i, np.random.randint(0, len(DISEASE_CLASSES))] = 1
        
        print(f"[v0] Sample data loaded: x_train={x_train.shape}, y_train={y_train.shape}")
        return (x_train, y_train), (x_val, y_val)
    
    def train_custom_cnn(self, x_train, y_train, x_val, y_val):
        """Train custom CNN model."""
        print("\n" + "="*50)
        print("Training Custom CNN Model")
        print("="*50)
        
        model = self.builder.build_custom_cnn(num_classes=len(DISEASE_CLASSES))
        model = self.builder.compile_model(model)
        
        print(model.summary())
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(MODELS_DIR, CUSTOM_CNN_MODEL)
        model.save(model_path)
        print(f"[v0] Custom CNN model saved to {model_path}")
        
        # Evaluate
        y_pred = model.predict(x_val)
        metrics = self.evaluator.calculate_metrics(y_val, y_pred, model_name='Custom CNN')
        self.evaluator.plot_training_history(history, model_name='Custom CNN')
        self.evaluator.plot_confusion_matrix(y_val, y_pred, model_name='Custom CNN')
        
        self.models_info['custom_cnn'] = metrics
        return model, history, metrics
    
    def train_mobilenetv2(self, x_train, y_train, x_val, y_val):
        """Train MobileNetV2 transfer learning model."""
        print("\n" + "="*50)
        print("Training MobileNetV2 Model")
        print("="*50)
        
        model = self.builder.build_mobilenetv2(num_classes=len(DISEASE_CLASSES))
        model = self.builder.compile_model(model)
        
        print(model.summary())
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(MODELS_DIR, MOBILENETV2_MODEL)
        model.save(model_path)
        print(f"[v0] MobileNetV2 model saved to {model_path}")
        
        # Evaluate
        y_pred = model.predict(x_val)
        metrics = self.evaluator.calculate_metrics(y_val, y_pred, model_name='MobileNetV2')
        self.evaluator.plot_training_history(history, model_name='MobileNetV2')
        self.evaluator.plot_confusion_matrix(y_val, y_pred, model_name='MobileNetV2')
        
        self.models_info['mobilenetv2'] = metrics
        return model, history, metrics
    
    def train_efficientnet(self, x_train, y_train, x_val, y_val):
        """Train EfficientNet transfer learning model."""
        print("\n" + "="*50)
        print("Training EfficientNet Model")
        print("="*50)
        
        model = self.builder.build_efficientnet(num_classes=len(DISEASE_CLASSES))
        model = self.builder.compile_model(model)
        
        print(model.summary())
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
        ]
        
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(MODELS_DIR, EFFICIENTNET_MODEL)
        model.save(model_path)
        print(f"[v0] EfficientNet model saved to {model_path}")
        
        # Evaluate
        y_pred = model.predict(x_val)
        metrics = self.evaluator.calculate_metrics(y_val, y_pred, model_name='EfficientNet')
        self.evaluator.plot_training_history(history, model_name='EfficientNet')
        self.evaluator.plot_confusion_matrix(y_val, y_pred, model_name='EfficientNet')
        
        self.models_info['efficientnet'] = metrics
        return model, history, metrics
    
    def compare_models(self):
        """Compare all trained models."""
        print("\n" + "="*50)
        print("Model Comparison")
        print("="*50)
        
        metrics_list = list(self.models_info.values())
        self.evaluator.plot_model_comparison(metrics_list)
        
        # Print comparison table
        print("\nModel Performance Comparison:")
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 68)
        for info in metrics_list:
            print(f"{info['model_name']:<20} {info['accuracy']:<12.4f} "
                  f"{info['precision']:<12.4f} {info['recall']:<12.4f} {info['f1_score']:<12.4f}")
        
        # Save comparison results
        comparison_data = {
            model: {
                'accuracy': float(info['accuracy']),
                'precision': float(info['precision']),
                'recall': float(info['recall']),
                'f1_score': float(info['f1_score'])
            }
            for model, info in self.models_info.items()
        }
        
        results_file = os.path.join(MODELS_DIR, 'model_comparison.json')
        with open(results_file, 'w') as f:
            json.dump(comparison_data, f, indent=4)
        
        print(f"\n[v0] Comparison results saved to {results_file}")
    
    def run(self):
        """Run the complete training pipeline."""
        print("\n" + "="*50)
        print("Skin Disease Detection Model Training")
        print("="*50)
        
        # Load data
        (x_train, y_train), (x_val, y_val) = self.load_sample_data()
        
        # Train models
        self.train_custom_cnn(x_train, y_train, x_val, y_val)
        self.train_mobilenetv2(x_train, y_train, x_val, y_val)
        self.train_efficientnet(x_train, y_train, x_val, y_val)
        
        # Compare models
        self.compare_models()
        
        print("\n" + "="*50)
        print("Training Complete!")
        print("="*50)

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()
