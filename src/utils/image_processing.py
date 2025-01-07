import cv2
import numpy as np
import tensorflow as tf
from google.cloud import vision
from google.cloud import storage
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Union

class ImageProcessor:
    def __init__(self, credentials_path: str = None):
        self.vision_client = vision.ImageAnnotatorClient() if credentials_path else None
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path."""
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for model input."""
        resized = cv2.resize(image, target_size)
        normalized = resized / 255.0
        return normalized
    
    def detect_objects(self, image_path: str) -> List[Dict]:
        """Detect objects using Google Cloud Vision."""
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        objects = self.vision_client.object_localization(image=image).localized_object_annotations
        return [{'name': obj.name, 'confidence': obj.score} for obj in objects]
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple]:
        """Detect faces using OpenCV."""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    def visualize_results(self, image: np.ndarray, detections: List[Dict] = None):
        """Visualize detection results."""
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        if detections:
            for det in detections:
                plt.text(10, 30, f"{det['name']}: {det['confidence']:.2f}", 
                        color='white', backgroundcolor='black')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    processor = ImageProcessor()
    # Example usage
    image = processor.load_image("sample.jpg")
    processed = processor.preprocess_image(image)
    enhanced = processor.enhance_image(image)
    faces = processor.detect_faces(image)