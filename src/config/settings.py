# settings.py

# Configuration settings for the machine learning diploma validator

# API keys for GCP services
GCP_VISION_API_KEY = 'your_gcp_vision_api_key'
GCP_AUTOML_API_KEY = 'your_gcp_automl_api_key'

# Model parameters
MODEL_NAME = 'diploma_classifier'
MODEL_VERSION = 'v1'
BATCH_SIZE = 32
EPOCHS = 50

# Paths for data storage
DATA_PATH = 'data/diplomas/'
MODEL_PATH = 'models/diploma_classifier.h5'
LOGS_PATH = 'logs/training_logs.txt'