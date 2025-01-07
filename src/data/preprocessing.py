import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List

def load_data(file_path: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load images and labels from directory structure.
    Expects subdirectories 'real' and 'fake' containing respective images.
    """
    images = []
    labels = []
    
    # Load real diplomas (label 1)
    real_path = os.path.join(file_path, 'real')
    for img_name in os.listdir(real_path):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(real_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(1)
    
    # Load fake diplomas (label 0)
    fake_path = os.path.join(file_path, 'fake')
    for img_name in os.listdir(fake_path):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(fake_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(0)
    
    return images, labels

def clean_data(images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Clean and standardize images:
    - Resize to standard dimensions
    - Convert to grayscale
    - Normalize pixel values
    """
    STANDARD_SIZE = (800, 600)  # Standard size for all images
    cleaned_images = []
    
    for img in images:
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize
        img = cv2.resize(img, STANDARD_SIZE)
        
        # Normalize pixel values to range [0,1]
        img = img.astype(np.float32) / 255.0
        
        cleaned_images.append(img)
    
    return cleaned_images

def split_data(images: List[np.ndarray], labels: List[int], 
               test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets
    """
    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Add channel dimension for CNN input
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def preprocess_pipeline(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline
    """
    # Load raw data
    images, labels = load_data(data_dir)
    
    # Clean images
    cleaned_images = clean_data(images)
    
    # Split into train/test sets
    return split_data(cleaned_images, labels)