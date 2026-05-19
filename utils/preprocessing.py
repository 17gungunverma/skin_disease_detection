import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import IMG_SIZE

class ImagePreprocessor:
    """Handles image preprocessing for skin lesion detection."""
    
    @staticmethod
    def load_and_preprocess_image(image_path):
        """Load and preprocess a single image."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values
            image = image.astype('float32') / 255.0
            
            return image
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    @staticmethod
    def load_and_preprocess_pil_image(pil_image):
        """Load and preprocess a PIL image."""
        try:
            # Convert PIL image to numpy array
            image = np.array(pil_image)
            
            # If grayscale, convert to RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # Resize to standard size
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values
            image = image.astype('float32') / 255.0
            
            return image
        except Exception as e:
            raise Exception(f"Error preprocessing PIL image: {str(e)}")
    
    @staticmethod
    def get_data_augmentation():
        """Get data augmentation generator."""
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    
    @staticmethod
    def get_validation_generator():
        """Get validation data generator (no augmentation)."""
        return ImageDataGenerator(rescale=1./255.)
