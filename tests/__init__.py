import os
import logging
from pathlib import Path

# Test configuration
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "test_data"
SAMPLE_IMAGES_DIR = TEST_DATA_DIR / "sample_images"

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
test_logger = logging.getLogger('tests')

# Test constants
TEST_IMAGE_SIZE = (300, 300)
TEST_BATCH_SIZE = 2
MOCK_CONFIDENCE_THRESHOLD = 0.5

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
SAMPLE_IMAGES_DIR.mkdir(exist_ok=True)

__all__ = [
    'TEST_DIR',
    'TEST_DATA_DIR',
    'SAMPLE_IMAGES_DIR',
    'test_logger',
    'TEST_IMAGE_SIZE',
    'TEST_BATCH_SIZE',
    'MOCK_CONFIDENCE_THRESHOLD'
]