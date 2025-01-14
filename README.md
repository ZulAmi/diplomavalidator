# Diploma Validation System 🎓

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Advanced diploma validation system using computer vision and machine learning to detect counterfeit educational certificates. The system analyzes security features, watermarks, and document authenticity through deep learning models.

## Features

- 🔍 Counterfeit detection using deep learning
- 🏛️ Institution-specific template matching
- 🔐 Security feature verification
- 📝 OCR for text validation
- 🎨 Image preprocessing and enhancement
- 📊 Confidence scoring system
- ⚡ Real-time validation pipeline

## Technologies

- Python 3.8+
- TensorFlow 2.8.0
- OpenCV 4.5.3
- Google Cloud Vision API
- scikit-learn 1.0.2
- NumPy/Pandas

## Project Structure
```
ml-diploma-validator
├── src
│   ├── models
│   │   └── classifier.py
│   ├── data
│   │   └── preprocessing.py
│   ├── utils
│   │   └── image_processing.py
│   └── config
│       └── settings.py
├── tests
│   └── test_classifier.py
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/diploma-validator.git
cd diploma-validator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. Configure your GCP settings in `src/config/settings.py`.

## Usage

```python
from diploma_validator import DiplomaValidator

validator = DiplomaValidator()
result = validator.validate("path/to/diploma.jpg")
print(f"Authenticity Score: {result.score}")
print(f"Security Features Detected: {result.security_features}")
```

## Testing
Run the unit tests to ensure everything is functioning correctly:
```
pytest tests/test_classifier.py
```

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.