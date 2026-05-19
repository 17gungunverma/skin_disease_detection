# import os
# import numpy as np
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import json

# from config import (
#     IMG_SIZE, BATCH_SIZE, EPOCHS, DISEASE_CLASSES, 
#     MODELS_DIR, DATA_DIR, CUSTOM_CNN_MODEL, 
#     MOBILENETV2_MODEL, EFFICIENTNET_MODEL, RANDOM_SEED
# )
# from utils.preprocessing import ImagePreprocessor
# from utils.model_builder import ModelBuilder
# from utils.evaluation import ModelEvaluator
# from utils.ham10000_loader import HAM10000DataLoader

# np.random.seed(RANDOM_SEED)

# class HAM10000ModelTrainer:
#     """Trains skin disease detection models using HAM10000 dataset."""
    
#     def __init__(self):
#         self.preprocessor = ImagePreprocessor()
#         self.builder = ModelBuilder()
#         self.evaluator = ModelEvaluator()
#         self.models_info = {}
#         self.class_weights = None
    
#     def verify_and_load_data(self):
#         """Verify HAM10000 dataset and load data."""
#         print("\n" + "="*60)
#         print("HAM10000 Dataset Verification")
#         print("="*60)
        
#         # Verify dataset
#         verification = HAM10000DataLoader.verify_dataset(data_dir=DATA_DIR)
        
#         print(f"Metadata file exists: {verification['metadata_exists']}")
#         print(f"Images directory exists: {verification['images_dir_exists']}")
#         print(f"Image count: {verification['image_count']}")
#         print(f"Metadata count: {verification['metadata_count']}")
#         print(f"Status: {verification['status']}")
        
#         if verification['status'] != 'Complete':
#             print("\n" + "!"*60)
#             print("ERROR: HAM10000 dataset is not properly set up!")
#             print("!"*60)
#             print("\nPlease follow the setup instructions in HAM10000_SETUP.md:")
#             print("  1. Create a Kaggle account and get API key")
#             print("  2. Download the dataset using Kaggle CLI")
#             print("  3. Extract images to data/HAM10000_images/")
#             print("  4. Verify HAM10000_metadata.csv exists in data/")
#             print("\nFor detailed instructions, see: HAM10000_SETUP.md")
#             raise ValueError("Dataset verification failed. See instructions above.")
        
#         print("\nDataset verification passed!")
        
#         # Load data
#         print("\n" + "="*60)
#         print("Loading HAM10000 Dataset")
#         print("="*60)
        
#         loader = HAM10000DataLoader(
#             data_dir=DATA_DIR,
#             metadata_file=os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
#         )
        
#         (x_train, y_train), (x_val, y_val) = loader.load_data(test_size=0.2)
#         self.class_weights = loader.get_class_weights()
        
#         return (x_train, y_train), (x_val, y_val)
    
#     def train_custom_cnn(self, x_train, y_train, x_val, y_val):
#         """Train custom CNN model with HAM10000 data."""
#         print("\n" + "="*60)
#         print("Training Custom CNN Model")
#         print("="*60)
        
#         model = self.builder.build_custom_cnn(num_classes=len(DISEASE_CLASSES))
#         model = self.builder.compile_model(model)
        
#         print(model.summary())
        
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
#         ]
        
#         print("\n[v0] Starting training with class weights...")
#         history = model.fit(
#             x_train, y_train,
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             validation_data=(x_val, y_val),
#             callbacks=callbacks,
#             class_weight=self.class_weights,
#             verbose=1
#         )
        
#         # Save model
#         model_path = os.path.join(MODELS_DIR, CUSTOM_CNN_MODEL)
#         model.save(model_path)
#         print(f"[v0] Custom CNN model saved to {model_path}")
        
#         # Evaluate
#         y_pred = model.predict(x_val)
#         metrics = self.evaluator.calculate_metrics(y_val, y_pred, model_name='Custom CNN')
#         self.evaluator.plot_training_history(history, model_name='Custom CNN')
#         self.evaluator.plot_confusion_matrix(y_val, y_pred, model_name='Custom CNN')
        
#         self.models_info['custom_cnn'] = metrics
#         return model, history, metrics
    
#     def train_mobilenetv2(self, x_train, y_train, x_val, y_val):
#         """Train MobileNetV2 transfer learning model with HAM10000 data."""
#         print("\n" + "="*60)
#         print("Training MobileNetV2 Model")
#         print("="*60)
        
#         model = self.builder.build_mobilenetv2(num_classes=len(DISEASE_CLASSES))
#         model = self.builder.compile_model(model)
        
#         print(model.summary())
        
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
#         ]
        
#         print("\n[v0] Starting training with class weights...")
#         history = model.fit(
#             x_train, y_train,
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             validation_data=(x_val, y_val),
#             callbacks=callbacks,
#             class_weight=self.class_weights,
#             verbose=1
#         )
        
#         # Save model
#         model_path = os.path.join(MODELS_DIR, MOBILENETV2_MODEL)
#         model.save(model_path)
#         print(f"[v0] MobileNetV2 model saved to {model_path}")
        
#         # Evaluate
#         y_pred = model.predict(x_val)
#         metrics = self.evaluator.calculate_metrics(y_val, y_pred, model_name='MobileNetV2')
#         self.evaluator.plot_training_history(history, model_name='MobileNetV2')
#         self.evaluator.plot_confusion_matrix(y_val, y_pred, model_name='MobileNetV2')
        
#         self.models_info['mobilenetv2'] = metrics
#         return model, history, metrics
    
#     def train_efficientnet(self, x_train, y_train, x_val, y_val):
#         """Train EfficientNet transfer learning model with HAM10000 data."""
#         print("\n" + "="*60)
#         print("Training EfficientNet Model")
#         print("="*60)
        
#         model = self.builder.build_efficientnet(num_classes=len(DISEASE_CLASSES))
#         model = self.builder.compile_model(model)
        
#         print(model.summary())
        
#         callbacks = [
#             EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
#         ]
        
#         print("\n[v0] Starting training with class weights...")
#         history = model.fit(
#             x_train, y_train,
#             batch_size=BATCH_SIZE,
#             epochs=EPOCHS,
#             validation_data=(x_val, y_val),
#             callbacks=callbacks,
#             class_weight=self.class_weights,
#             verbose=1
#         )
        
#         # Save model
#         model_path = os.path.join(MODELS_DIR, EFFICIENTNET_MODEL)
#         model.save(model_path)
#         print(f"[v0] EfficientNet model saved to {model_path}")
        
#         # Evaluate
#         y_pred = model.predict(x_val)
#         metrics = self.evaluator.calculate_metrics(y_val, y_pred, model_name='EfficientNet')
#         self.evaluator.plot_training_history(history, model_name='EfficientNet')
#         self.evaluator.plot_confusion_matrix(y_val, y_pred, model_name='EfficientNet')
        
#         self.models_info['efficientnet'] = metrics
#         return model, history, metrics
    
#     def compare_models(self):
#         """Compare all trained models."""
#         print("\n" + "="*60)
#         print("Model Comparison")
#         print("="*60)
        
#         metrics_list = list(self.models_info.values())
#         self.evaluator.plot_model_comparison(metrics_list)
        
#         # Print comparison table
#         print("\nModel Performance Comparison:")
#         print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
#         print("-" * 68)
#         for info in metrics_list:
#             print(f"{info['model_name']:<20} {info['accuracy']:<12.4f} "
#                   f"{info['precision']:<12.4f} {info['recall']:<12.4f} {info['f1_score']:<12.4f}")
        
#         # Save comparison results
#         comparison_data = {
#             model: {
#                 'accuracy': float(info['accuracy']),
#                 'precision': float(info['precision']),
#                 'recall': float(info['recall']),
#                 'f1_score': float(info['f1_score'])
#             }
#             for model, info in self.models_info.items()
#         }
        
#         results_file = os.path.join(MODELS_DIR, 'model_comparison.json')
#         with open(results_file, 'w') as f:
#             json.dump(comparison_data, f, indent=4)
        
#         print(f"\n[v0] Comparison results saved to {results_file}")
    
#     def run(self):
#         """Run the complete training pipeline with HAM10000 data."""
#         print("\n" + "="*60)
#         print("Skin Disease Detection Training - HAM10000 Dataset")
#         print("="*60)
        
#         # Verify and load data
#         (x_train, y_train), (x_val, y_val) = self.verify_and_load_data()
        
#         # Train models
#         self.train_custom_cnn(x_train, y_train, x_val, y_val)
#         self.train_mobilenetv2(x_train, y_train, x_val, y_val)
#         self.train_efficientnet(x_train, y_train, x_val, y_val)
        
#         # Compare models
#         self.compare_models()
        
#         print("\n" + "="*60)
#         print("Training Complete!")
#         print("="*60)
#         print("\nNext steps:")
#         print("  1. Start the Flask web app: python app.py")
#         print("  2. Open http://localhost:5000 in your browser")
#         print("  3. Upload skin lesion images to test predictions")
#         print("\nTrained models saved to:")
#         print(f"  - {os.path.join(MODELS_DIR, CUSTOM_CNN_MODEL)}")
#         print(f"  - {os.path.join(MODELS_DIR, MOBILENETV2_MODEL)}")
#         print(f"  - {os.path.join(MODELS_DIR, EFFICIENTNET_MODEL)}")

# if __name__ == '__main__':
#     trainer = HAM10000ModelTrainer()
#     trainer.run()







import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

from config import (
    IMG_SIZE, BATCH_SIZE, EPOCHS, DISEASE_CLASSES,
    MODELS_DIR, DATA_DIR, CUSTOM_CNN_MODEL,
    MOBILENETV2_MODEL, EFFICIENTNET_MODEL, RANDOM_SEED
)
from utils.model_builder import ModelBuilder
from utils.evaluation import ModelEvaluator
from utils.ham10000_loader import HAM10000DataLoader

np.random.seed(RANDOM_SEED)


class HAM10000ModelTrainer:
    """Trains skin disease detection models using HAM10000 dataset."""

    def __init__(self):
        self.builder       = ModelBuilder()
        self.evaluator     = ModelEvaluator()
        self.models_info   = {}
        self.class_weights = None
        # generators & encoder are stored after verify_and_load_data()
        self.train_gen     = None
        self.val_gen       = None
        self.label_encoder = None

    # ──────────────────────────────────────────────────────────────────────────
    def verify_and_load_data(self):
        """Verify HAM10000 dataset and return train/val generators."""
        print("\n" + "=" * 60)
        print("HAM10000 Dataset Verification")
        print("=" * 60)

        verification = HAM10000DataLoader.verify_dataset(data_dir=DATA_DIR)

        print(f"Metadata file exists: {verification['metadata_exists']}")
        print(f"Images directory exists: {verification['images_dir_exists']}")
        print(f"Image count: {verification['image_count']}")
        print(f"Metadata count: {verification['metadata_count']}")
        print(f"Status: {verification['status']}")

        if verification['status'] != 'Complete':
            print("\n" + "!" * 60)
            print("ERROR: HAM10000 dataset is not properly set up!")
            print("!" * 60)
            print("\nPlease follow the setup instructions in HAM10000_SETUP.md:")
            print("  1. Create a Kaggle account and get API key")
            print("  2. Download the dataset using Kaggle CLI")
            print("  3. Extract images to data/HAM10000_images/")
            print("  4. Verify HAM10000_metadata.csv exists in data/")
            raise ValueError("Dataset verification failed. See instructions above.")

        print("\nDataset verification passed!")

        # ── load data ─────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("Loading HAM10000 Dataset")
        print("=" * 60)

        loader = HAM10000DataLoader(
            data_dir=DATA_DIR,
            metadata_file=os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
        )

        #  NEW: returns generators, not numpy arrays
        train_gen, val_gen, label_encoder = loader.load_data(
            test_size=0.2,
            batch_size=BATCH_SIZE
        )

        self.class_weights = loader.get_class_weights()
        self.train_gen     = train_gen
        self.val_gen       = val_gen
        self.label_encoder = label_encoder

        return train_gen, val_gen

    # ──────────────────────────────────────────────────────────────────────────
    def _get_callbacks(self):
        """Return standard Keras callbacks."""
        return [
            EarlyStopping(
                monitor='val_loss', patience=5,
                restore_best_weights=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=3, verbose=1
            )
        ]

    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_model(self, model, val_gen, model_name):
        """Run predictions on val generator and compute metrics."""
        print(f"\n[v0] Evaluating {model_name}...")
        y_pred = model.predict(val_gen, verbose=1)

        # Collect true labels from generator
        y_true = np.concatenate([val_gen[i][1] for i in range(len(val_gen))], axis=0)

        metrics = self.evaluator.calculate_metrics(y_true, y_pred, model_name=model_name)
        self.evaluator.plot_training_history   # called per-model below
        self.evaluator.plot_confusion_matrix(y_true, y_pred, model_name=model_name)
        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    def train_custom_cnn(self, train_gen, val_gen):
        """Train custom CNN model with HAM10000 data."""
        print("\n" + "=" * 60)
        print("Training Custom CNN Model")
        print("=" * 60)

        model = self.builder.build_custom_cnn(num_classes=len(DISEASE_CLASSES))
        model = self.builder.compile_model(model)
        print(model.summary())

        print("\n[v0] Starting training with class weights...")
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=self._get_callbacks(),
            class_weight=self.class_weights,
            verbose=1
        )

        model_path = os.path.join(MODELS_DIR, CUSTOM_CNN_MODEL)
        model.save(model_path)
        print(f"[v0] Custom CNN model saved to {model_path}")

        self.evaluator.plot_training_history(history, model_name='Custom CNN')
        metrics = self._evaluate_model(model, val_gen, model_name='Custom CNN')
        self.models_info['custom_cnn'] = metrics
        return model, history, metrics

    # ──────────────────────────────────────────────────────────────────────────
    def train_mobilenetv2(self, train_gen, val_gen):
        """Train MobileNetV2 transfer learning model with HAM10000 data."""
        print("\n" + "=" * 60)
        print("Training MobileNetV2 Model")
        print("=" * 60)

        model = self.builder.build_mobilenetv2(num_classes=len(DISEASE_CLASSES))
        model = self.builder.compile_model(model)
        print(model.summary())

        print("\n[v0] Starting training with class weights...")
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=self._get_callbacks(),
            class_weight=self.class_weights,
            verbose=1
        )

        model_path = os.path.join(MODELS_DIR, MOBILENETV2_MODEL)
        model.save(model_path)
        print(f"[v0] MobileNetV2 model saved to {model_path}")

        self.evaluator.plot_training_history(history, model_name='MobileNetV2')
        metrics = self._evaluate_model(model, val_gen, model_name='MobileNetV2')
        self.models_info['mobilenetv2'] = metrics
        return model, history, metrics

    # ──────────────────────────────────────────────────────────────────────────
    def train_efficientnet(self, train_gen, val_gen):
        """Train EfficientNet transfer learning model with HAM10000 data."""
        print("\n" + "=" * 60)
        print("Training EfficientNet Model")
        print("=" * 60)

        model = self.builder.build_efficientnet(num_classes=len(DISEASE_CLASSES))
        model = self.builder.compile_model(model)
        print(model.summary())

        print("\n[v0] Starting training with class weights...")
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=self._get_callbacks(),
            class_weight=self.class_weights,
            verbose=1
        )

        model_path = os.path.join(MODELS_DIR, EFFICIENTNET_MODEL)
        model.save(model_path)
        print(f"[v0] EfficientNet model saved to {model_path}")

        self.evaluator.plot_training_history(history, model_name='EfficientNet')
        metrics = self._evaluate_model(model, val_gen, model_name='EfficientNet')
        self.models_info['efficientnet'] = metrics
        return model, history, metrics

    # ──────────────────────────────────────────────────────────────────────────
    def compare_models(self):
        """Compare all trained models."""
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)

        metrics_list = list(self.models_info.values())
        self.evaluator.plot_model_comparison(metrics_list)

        print("\nModel Performance Comparison:")
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 68)
        for info in metrics_list:
            print(
                f"{info['model_name']:<20} {info['accuracy']:<12.4f} "
                f"{info['precision']:<12.4f} {info['recall']:<12.4f} "
                f"{info['f1_score']:<12.4f}"
            )

        comparison_data = {
            model: {
                'accuracy' : float(info['accuracy']),
                'precision': float(info['precision']),
                'recall'   : float(info['recall']),
                'f1_score' : float(info['f1_score'])
            }
            for model, info in self.models_info.items()
        }

        os.makedirs(MODELS_DIR, exist_ok=True)
        results_file = os.path.join(MODELS_DIR, 'model_comparison.json')
        with open(results_file, 'w') as f:
            json.dump(comparison_data, f, indent=4)

        print(f"\n[v0] Comparison results saved to {results_file}")

    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        """Run the complete training pipeline with HAM10000 data."""
        print("\n" + "=" * 60)
        print("Skin Disease Detection Training - HAM10000 Dataset")
        print("=" * 60)

        #  returns generators now (not numpy arrays)
        train_gen, val_gen = self.verify_and_load_data()

        #  pass generators directly to each trainer
        self.train_custom_cnn  (train_gen, val_gen)
        self.train_mobilenetv2 (train_gen, val_gen)
        self.train_efficientnet(train_gen, val_gen)

        self.compare_models()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Start the Flask web app: python app.py")
        print("  2. Open http://localhost:5000 in your browser")
        print("  3. Upload skin lesion images to test predictions")
        print("\nTrained models saved to:")
        print(f"  - {os.path.join(MODELS_DIR, CUSTOM_CNN_MODEL)}")
        print(f"  - {os.path.join(MODELS_DIR, MOBILENETV2_MODEL)}")
        print(f"  - {os.path.join(MODELS_DIR, EFFICIENTNET_MODEL)}")


if __name__ == '__main__':
    trainer = HAM10000ModelTrainer()
    trainer.run()