# MLOps Linear Regression Pipeline

A modular, production-ready MLOps pipeline using Linear Regression on the California Housing dataset — featuring training, testing, 8-bit quantization, Docker containerization, and GitHub Actions CI/CD automation.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Folder Layout](#folder-layout)
- [Install & Run](#install--run)
- [Model Training](#model-training)
- [Quantization](#quantization)
- [Predictions](#predictions)
- [Model Metrics](#model-metrics)
- [Docker Deployment](#docker-deployment)
- [CI/CD Workflow](#cicd-workflow)
- [Testing Framework](#testing-framework)
- [Quantization Logic](#quantization-logic)

---

## Overview

This project delivers a complete MLOps-ready pipeline with:

- **Training**: scikit-learn Linear Regression
- **Evaluation**: R² and MSE tracking
- **Quantization**: Manual 8-bit compression of model weights
- **Packaging**: Docker for deployment-ready images
- **Automation**: GitHub Actions for CI/CD and model promotion
- **Monitoring**: Sample predictions and diff analysis

---

## Environment Setup

```powershell
# Create virtual environment
python -m venv env
.\env\Scripts\activate

# Install required libraries
pip install -r requirements.txt


** Folder Layout**

mlops-linear-regression/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── quantize.py
│   ├── utils.py
│   └── __init__.py
│
├── tests/
│   ├── test_train.py
│   └── __init__.py
│
├── models/
│   ├── linear_model.joblib
│   └── quant_params.joblib
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md


**Install & Run**

# Clone the project
git clone https://github.com/<your-username>/mlops-linear-regression.git
cd mlops-linear-regression

# Setup environment
python -m venv venv
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

**Model Training**

Fetching California Housing data...
Initializing Linear Regression...
Fitting model...
Training Complete.
Model R² Score: 0.6042
MSE (Loss): 0.5117
Model saved to models/linear_model.joblib

**Quantization**

python quantize.py


**Output (Sample)**

Loading model...
Original Coefficients: [0.48, 0.01, -0.11, 0.81, ...]
Original Intercept: -36.0189

Scaling weights to 8-bit range...
Intercept quantized.
Saved quantized parameters to models/quant_params.joblib

Max prediction error from dequantization: 0.0000017
Mean difference: 0.0000015
Quantization successful

** Predictions**

python predict.py

**Sample Output**

Predicting on test dataset...

Prediction Results (first 5):
Ground Truth: 0.58 | Predicted: 0.74 | Diff: 0.16
Ground Truth: 1.52 | Predicted: 1.79 | Diff: 0.27
Ground Truth: 4.98 | Predicted: 2.74 | Diff: 2.24
Ground Truth: 2.20 | Predicted: 2.82 | Diff: 0.62
Ground Truth: 2.85 | Predicted: 2.66 | Diff: 0.19

R² Score: 0.6042
MSE: 0.5117


**Model Metrics**

| Metric               | Base Model | Quantized Model | Difference  |
| -------------------- | ---------- | --------------- | ----------- |
| **R² Score**         | 0.6042     | 0.6042          | 0.0000      |
| **MSE**              | 0.5117     | 0.5117          | 0.0000      |
| **Max Error (abs)**  | —          | 0.0000017       | ↑ 0.0000017 |
| **Mean Error (abs)** | —          | 0.0000015       | ↑ 0.0000015 |
| **File Size (KB)**   | 1.1 KB     | 0.3 KB          | ↓ 0.8 KB    |


**Docker Deployment
Build the Image**

docker build -t mlops-lr .

**Run the Container**

Model loaded from disk.
Test data loaded.
Generating predictions...

Top 5 Predictions:
True: 3.25 | Predicted: 3.01 | Diff: 0.24
...

Model R² Score: 0.6042
MSE: 0.5117

**CI/CD Workflow**
The GitHub Actions workflow is triggered on every push to master:

Workflow Steps:
Unit Testing
 Runs pytest on training pipeline
 Asserts model performance and saves metrics

Model Training & Quantization:-
 Trains model
 Runs quantization
 Uploads artifacts

Docker Build:-
 Builds image using Dockerfile
 Verifies prediction step inside container


**Testing Framework**
# Run tests
python -m pytest tests/ -v

# Coverage Report
pytest --cov=src --cov-report=term

**Tests Covered**

Dataset shape and structure
Linear model coefficient validation
Training success with R² > 0.55
Model persistence (joblib file check)


**Quantization Logic
Overview**
We implement custom 8-bit quantization using manual coefficient scaling. This compresses the model size with negligible impact on accuracy.

**Formula**

# Quantize
scaled = coef * scale_factor
q = ((scaled - min_val) / (max_val - min_val)) * 255
quantized = q.astype(np.uint8)

# Dequantize
deq = (quantized / 255) * (max_val - min_val) + min_val
original = deq / scale_factor

**Summary**
This pipeline ensures a fully reproducible, testable, and deployable machine learning workflow, suitable for enterprise-grade CI/CD and edge deployment.

