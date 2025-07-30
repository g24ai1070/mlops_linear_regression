import numpy as np
import joblib
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def load_dataset():
    """Load and split the California Housing dataset into training and test sets."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_model():
    """Initialize a LinearRegression model."""
    return LinearRegression()


def save_model(model, filepath):
    """Save a model to the specified filepath using joblib."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load a model from a file using joblib."""
    return joblib.load(filepath)


def calculate_metrics(y_true, y_pred):
    """Calculate R² and MSE metrics."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse


def quantize_to_uint8(values, scale_factor=None):
    """Quantize float values to uint8 using global scaling."""
    if np.all(values == 0):
        return np.zeros(values.shape, dtype=np.uint8), 0.0, 0.0, 1.0

    if scale_factor is None:
        abs_max = np.abs(values).max()
        scale_factor = 200.0 / abs_max if abs_max > 0 else 1.0

    scaled = values * scale_factor
    min_val, max_val = scaled.min(), scaled.max()

    if max_val == min_val:
        quantized = np.full(values.shape, 127, dtype=np.uint8)
        return quantized, min_val, max_val, scale_factor

    normalized = ((scaled - min_val) / (max_val - min_val)) * 255
    quantized = np.clip(normalized, 0, 255).astype(np.uint8)

    return quantized, min_val, max_val, scale_factor


def dequantize_from_uint8(quantized_values, min_val, max_val, scale_factor):
    """Reconstruct original float values from uint8 quantized data."""
    if max_val == min_val:
        return np.full(quantized_values.shape, min_val / scale_factor, dtype=np.float32)

    value_range = max_val - min_val
    denormalized = (quantized_values.astype(np.float32) / 255.0) * value_range + min_val
    return denormalized / scale_factor


def quantize_to_uint8_individual(values):
    """Quantize each float individually to uint8 with metadata for dequantization."""
    quantized = np.zeros(values.shape, dtype=np.uint8)
    metadata = []

    for i, val in enumerate(values):
        if val == 0:
            quantized[i] = 127
            metadata.append({'min_val': 0.0, 'max_val': 0.0, 'scale': 1.0})
        else:
            abs_val = abs(val)
            scale_factor = 127.0 / abs_val
            quantized_val = 127 - abs_val * scale_factor if val < 0 else 128 + abs_val * scale_factor
            quantized[i] = int(np.clip(quantized_val, 0, 255))
            metadata.append({
                'min_val': val,
                'max_val': val,
                'scale': scale_factor,
                'original': val
            })

    return quantized, metadata


def dequantize_from_uint8_individual(quantized_values, metadata):
    """Dequantize uint8 values back to float using stored metadata."""
    dequantized = np.zeros(quantized_values.shape, dtype=np.float32)

    for i, (q, meta) in enumerate(zip(quantized_values, metadata)):
        if meta['scale'] == 1.0:
            dequantized[i] = 0.0
        elif q <= 127:
            dequantized[i] = -(127 - q) / meta['scale']
        else:
            dequantized[i] = (q - 128) / meta['scale']

    return dequantized
