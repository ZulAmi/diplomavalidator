import unittest
from src.models.classifier import DiplomaClassifier
import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import os
from src.utils.image_processing import ImageProcessor

class TestDiplomaClassifier(unittest.TestCase):

    def setUp(self):
        self.classifier = DiplomaClassifier()

    def test_train(self):
        # Assuming we have a mock dataset
        mock_data = [...]  # Replace with actual mock data
        self.classifier.train(mock_data)
        self.assertIsNotNone(self.classifier.model)

    def test_predict(self):
        # Assuming we have a mock image
        mock_image = ...  # Replace with actual mock image
        prediction = self.classifier.predict(mock_image)
        self.assertIn(prediction, ['real', 'fake'])

    def test_evaluate(self):
        # Assuming we have a mock dataset and labels
        mock_data = [...]  # Replace with actual mock data
        mock_labels = [...]  # Replace with actual mock labels
        accuracy = self.classifier.evaluate(mock_data, mock_labels)
        self.assertGreaterEqual(accuracy, 0.5)  # Assuming 50% is the baseline

@pytest.fixture
def processor():
    return ImageProcessor()

@pytest.fixture
def sample_image():
    # Create a simple test image
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def mock_vision_client():
    with patch('google.cloud.vision.ImageAnnotatorClient') as mock:
        yield mock

def test_load_image(processor, tmp_path):
    # Create a temporary test image
    test_image_path = tmp_path / "test.jpg"
    cv2.imwrite(str(test_image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    
    loaded_image = processor.load_image(str(test_image_path))
    assert isinstance(loaded_image, np.ndarray)
    assert loaded_image.shape[-1] == 3  # RGB channels

def test_preprocess_image(processor, sample_image):
    target_size = (224, 224)
    processed = processor.preprocess_image(sample_image, target_size)
    
    assert processed.shape[:2] == target_size
    assert processed.max() <= 1.0
    assert processed.min() >= 0.0

def test_detect_faces(processor, sample_image):
    # Create an image with a face-like pattern
    face_image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(face_image, (150, 150), 50, (255, 255, 255), -1)
    
    faces = processor.detect_faces(face_image)
    assert isinstance(faces, (list, tuple))

def test_enhance_image(processor, sample_image):
    enhanced = processor.enhance_image(sample_image)
    
    assert enhanced.shape == sample_image.shape
    assert isinstance(enhanced, np.ndarray)
    assert enhanced.dtype == sample_image.dtype

@pytest.mark.integration
def test_detect_objects(processor, mock_vision_client, tmp_path):
    # Mock the Vision API response
    mock_response = Mock()
    mock_response.localized_object_annotations = [
        Mock(name="cat", score=0.95),
        Mock(name="dog", score=0.85)
    ]
    mock_vision_client.return_value.object_localization.return_value = mock_response
    
    # Create test image
    test_image_path = tmp_path / "test_object.jpg"
    cv2.imwrite(str(test_image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    
    results = processor.detect_objects(str(test_image_path))
    assert len(results) == 2
    assert results[0]['name'] == "cat"
    assert results[0]['confidence'] == 0.95

@pytest.mark.performance
def test_processing_performance(processor, sample_image):
    import time
    
    start_time = time.time()
    processor.preprocess_image(sample_image)
    processing_time = time.time() - start_time
    
    assert processing_time < 1.0  # Should process in less than 1 second

def test_visualization(processor, sample_image):
    import matplotlib.pyplot as plt
    
    with patch.object(plt, 'show') as mock_show:
        detections = [{'name': 'test', 'confidence': 0.9}]
        processor.visualize_results(sample_image, detections)
        mock_show.assert_called_once()

def test_with_dummy_images(processor, test_data_dir):
    # Test with first dummy image
    image_path = str(test_data_dir / "test1.jpg")
    image = processor.load_image(image_path)
    assert image is not None
    
    # Process image
    processed = processor.preprocess_image(image)
    assert processed.shape == (224, 224, 3)
    
    # Test face detection
    faces = processor.detect_faces(image)
    assert isinstance(faces, (list, tuple))
    
    # Test enhancement
    enhanced = processor.enhance_image(image)
    assert enhanced.shape == image.shape

if __name__ == '__main__':
    unittest.main()
    pytest.main([__file__, '-v'])