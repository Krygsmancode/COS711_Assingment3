import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

class MalariaPredictor:
    def __init__(self):
        """Initialize the predictor."""
        self.model = None
        self.input_shape = None
        self.class_labels = ['NEG', 'Trophozoite', 'WBC']
    
    def load_model(self, model_path):
        """
        Load a saved TensorFlow model.
        
        Args:
            model_path (str): Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.input_shape = self.model.input_shape[1:]  # Remove batch dimension
            print(f"Model loaded successfully. Input shape: {self.input_shape}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def preprocess_image(self, image, color_format='BGR'):
        """
        Preprocess the input image to match model requirements.
        
        Args:
            image: Can be either:
                  - str: Path to the input image
                  - numpy.ndarray: Image array (BGR or RGB format)
                  - PIL.Image: PIL Image object
            color_format (str): Color format of the input image ('BGR' or 'RGB')
                              Only applicable when input is numpy array
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Handle different input types
            if isinstance(image, str):
                # Load image from file path using cv2
                img = cv2.imread(str(image))
                if img is None:
                    raise ValueError(f"Could not load image: {image}")
                color_format = 'BGR'  # cv2.imread loads in BGR
                
            elif isinstance(image, np.ndarray):
                img = image.copy()
                
            elif isinstance(image, Image.Image):
                # Convert PIL Image to numpy array (RGB format)
                img = np.array(image)
                color_format = 'RGB'
                
            else:
                raise ValueError("Unsupported image type. Must be file path, numpy array, or PIL Image")
            
            # Convert color format if necessary
            if color_format.upper() == 'RGB' and self.model.input_shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Resize to match input shape
            if img.shape[:2] != self.input_shape[:2]:
                img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            
            # Convert to float16 and normalize
            img = img.astype(np.float16) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def predict(self, image, color_format='BGR', show_progress_bar=True):
        """
        Make a prediction on an input image.
        
        Args:
            image: Can be either:
                  - str: Path to the input image
                  - numpy.ndarray: Image array (BGR or RGB format)
                  - PIL.Image: PIL Image object
            color_format (str): Color format of the input image ('BGR' or 'RGB')
                              Only applicable when input is numpy array
            
        Returns:
            dict: Prediction results including class, confidence, and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        try:
            # Preprocess the image
            processed_image = self.preprocess_image(image, color_format)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            confidence_score = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_labels[predicted_class_idx]
            
            return {
                'class': predicted_class,
                'confidence': confidence_score,
                'probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.class_labels, predictions[0])
                }
            }
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")

    def batch_predict(self, images, color_format='BGR'):
        """
        Make predictions on multiple images.
        
        Args:
            images (list): List of images (can be mix of file paths and image objects)
            color_format (str): Color format of the input images ('BGR' or 'RGB')
                              Only applicable when input is numpy array
            
        Returns:
            list: List of prediction dictionaries
        """
        return [self.predict(image, color_format) for image in tqdm(images, desc="Processing images")]