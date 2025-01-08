import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    
    # Google Cloud Vision settings
    GCP_CREDENTIALS_PATH: Optional[str] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Image processing parameters
    TARGET_IMAGE_SIZE: tuple = (224, 224)
    BATCH_SIZE: int = 32
    
    # Model parameters
    MODEL_VERSION: str = "v1.0"
    CONFIDENCE_THRESHOLD: float = 0.75
    
    def __post_init__(self):
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        
        # Validate GCP credentials
        if not self.GCP_CREDENTIALS_PATH:
            raise ValueError("Google Cloud credentials not found in environment variables")

# Create singleton instance
config = Config()

__all__ = ['config']