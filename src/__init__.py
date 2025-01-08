import logging
from pathlib import Path
from .config import config

# Package metadata
__version__ = '1.0.0'
__author__ = 'Your Name'
__license__ = 'MIT'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Package root directory
PACKAGE_ROOT = Path(__file__).parent

# Export main components
from .image_processing import ImageProcessor

__all__ = [
    'ImageProcessor',
    'config',
    '__version__',
    'logger',
    'PACKAGE_ROOT'
]