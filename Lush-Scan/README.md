# Lush-Scan

A machine learning project for detecting deforestation in satellite imagery using multiple image processing techniques and Random Forest classification.

## Features

- **Multiple Image Processing Techniques:**
  - Gaussian Blur
  - Median Filtering
  - Adaptive Thresholding
  - Edge Detection (Canny and Laplacian)
  - Enhanced Segmentation using HSV and LAB color spaces

- **Comprehensive Feature Extraction:**
  - Color Statistics
  - Texture Analysis
  - Color Histograms
  - Vegetation Ratio Calculation

- **Machine Learning:**
  - Random Forest Classification
  - Feature Importance Analysis
  - Model Performance Evaluation
  - Model Persistence

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Pree46/Deforestation-Detection.git
   cd Deforestation-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:
   ```bash
   python enhanced_deforestation_detection.py
   ```

2. The script will:
   - Process all images with multiple techniques
   - Extract comprehensive features
   - Train a Random Forest model
   - Display evaluation metrics
   - Show visualizations of the processing steps
   - Save the trained model

## Results

The model provides:
- Classification accuracy scores
- Detailed classification reports
- Feature importance rankings
- Visualizations of image processing steps
- Saved model for future predictions

## Dependencies

Core requirements:
- numpy
- opencv-python
- scikit-learn
- matplotlib
- pandas
- joblib
- requests
- tqdm

## Author

R Preethi

## License

This project is licensed under the MIT License - see the LICENSE file for details.


