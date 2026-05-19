import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, auc
)
from config import DISEASE_CLASSES, RESULTS_DIR
import os

class ModelEvaluator:
    """Evaluates and visualizes model performance."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba=None, model_name='Model'):
        """Calculate evaluation metrics."""
        # Convert one-hot to class labels if needed
        if len(y_true.shape) > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true
        
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true_labels, y_pred_labels),
            'precision': precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'recall': recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true_labels, y_pred_labels)
        }
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name='Model'):
        """Plot confusion matrix."""
        if len(y_true.shape) > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true
        
        y_pred_labels = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true_labels, y_pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=list(DISEASE_CLASSES.values()),
                    yticklabels=list(DISEASE_CLASSES.values()))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}.png')
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    @staticmethod
    def plot_training_history(history, model_name='Model'):
        """Plot training and validation loss/accuracy."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'Model Accuracy - {model_name}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'Model Loss - {model_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(RESULTS_DIR, f'training_history_{model_name}.png')
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    @staticmethod
    def plot_model_comparison(metrics_list):
        """Compare multiple models."""
        model_names = [m['model_name'] for m in metrics_list]
        accuracies = [m['accuracy'] for m in metrics_list]
        precisions = [m['precision'] for m in metrics_list]
        recalls = [m['recall'] for m in metrics_list]
        f1_scores = [m['f1_score'] for m in metrics_list]
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filepath = os.path.join(RESULTS_DIR, 'model_comparison.png')
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    @staticmethod
    def get_classification_report(y_true, y_pred):
        """Get detailed classification report."""
        if len(y_true.shape) > 1:
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_true_labels = y_true
        
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        report = classification_report(
            y_true_labels, y_pred_labels,
            target_names=list(DISEASE_CLASSES.values()),
            output_dict=True
        )
        
        return report
