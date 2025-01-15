FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpango1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials.json

# Default command to run tests
CMD ["pytest", "tests/", "-v", "--cov=src"]