import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-diploma-validator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="ML-based system for validating educational certificates and diplomas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-diploma-validator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow==2.8.0",
        "scikit-learn==1.0.2",
        "opencv-python==4.5.3.20210927",
        "google-cloud-vision==2.5.0",
        "google-cloud-storage==1.42.3",
        "pandas==1.3.5",
        "numpy==1.21.4",
        "matplotlib==3.5.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "black>=21.12b0",
            "isort>=5.10.1",
            "flake8>=4.0.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "validate-diploma=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml"],
    },
)