import pytest
import os
from src.image_processing import ImageProcessor
from pathlib import Path

class TestDiplomaValidation:
    @pytest.fixture
    def test_data_path(self):
        return Path(__file__).parent / "test_data" / "diplomas"
    
    @pytest.fixture
    def processor(self):
        return ImageProcessor()
    
    def test_real_diploma_detection(self, processor, test_data_path):
        real_diploma_path = test_data_path / "real" / "sample_real.jpg"
        result = processor.validate_diploma(str(real_diploma_path))
        assert result['authenticity_score'] > 0.8
        assert result['security_features_detected'] >= 3
    
    def test_fake_diploma_detection(self, processor, test_data_path):
        fake_diploma_path = test_data_path / "fake" / "sample_fake.jpg"
        result = processor.validate_diploma(str(fake_diploma_path))
        assert result['authenticity_score'] < 0.5