import os
import numpy as np
import joblib
from utils import (
    load_model,
    load_dataset,
    quantize_to_uint8,
    dequantize_from_uint8,
    quantize_to_uint8_individual,
    dequantize_from_uint8_individual
)


def save_params(model, save_dir="models"):
    """Extract and save raw model parameters."""
    coef = model.coef_
    intercept = model.intercept_

    os.makedirs(save_dir, exist_ok=True)

    raw_params = {
        "coef": coef,
        "intercept": intercept
    }
    joblib.dump(raw_params, os.path.join(save_dir, "unquant_params.joblib"))
    print(" Raw model parameters saved.")

    return coef, intercept


def quantize_params(coef, intercept, save_dir="models"):
    """Quantize model parameters and save them with metadata."""
    print(" Quantizing coefficients...")
    quant_coef, coef_metadata = quantize_to_uint8_individual(coef)

    print(f" Quantizing intercept: {intercept:.6f}")
    quant_intercept, int_min, int_max, int_scale = quantize_to_uint8(np.array([intercept]))

    quant_params = {
        "quant_coef": quant_coef,
        "coef_metadata": coef_metadata,
        "quant_intercept": quant_intercept[0],
        "int_min": int_min,
        "int_max": int_max,
        "int_scale": int_scale
    }

    joblib.dump(quant_params, os.path.join(save_dir, "quant_params.joblib"))
    print(" Quantized parameters saved.")

    return quant_params


def evaluate_dequantization(coef, intercept, quant_params):
    """Check dequantized parameters accuracy."""
    dequant_coef = dequantize_from_uint8_individual(
        quant_params["quant_coef"],
        quant_params["coef_metadata"]
    )

    dequant_intercept = dequantize_from_uint8(
        np.array([quant_params["quant_intercept"]]),
        quant_params["int_min"],
        quant_params["int_max"],
        quant_params["int_scale"]
    )[0]

    coef_error = np.abs(coef - dequant_coef).max()
    intercept_error = np.abs(intercept - dequant_intercept)

    print(f"📏 Max coefficient error: {coef_error:.6f}")
    print(f"📏 Intercept error: {intercept_error:.6f}")

    return dequant_coef, dequant_intercept


def compare_predictions(model, coef, intercept, dequant_coef, dequant_intercept):
    """Run inference on test data and compare predictions."""
    _, X_test, _, y_test = load_dataset()

    print("\n Running inference comparison (first 5 samples)...")
    sklearn_pred = model.predict(X_test[:5])
    manual_pred_original = X_test[:5] @ coef + intercept
    manual_pred_dequant = X_test[:5] @ dequant_coef + dequant_intercept

    diff_sklearn_vs_manual = np.abs(sklearn_pred - manual_pred_original)
    diff_manual_vs_dequant = np.abs(manual_pred_original - manual_pred_dequant)
    diff_sklearn_vs_dequant = np.abs(sklearn_pred - manual_pred_dequant)

    print(" Predictions:")
    for i in range(5):
        print(f"Sample {i + 1}: Sklearn={sklearn_pred[i]:.4f} | ManualOrig={manual_pred_original[i]:.4f} | Dequant={manual_pred_dequant[i]:.4f}")

    print("\n📉 Differences:")
    print(f"Sklearn vs Manual:   {diff_sklearn_vs_manual}")
    print(f"Original vs Dequant: {diff_manual_vs_dequant}")
    print(f"Sklearn vs Dequant:  {diff_sklearn_vs_dequant}")
    print(f"Max difference:      {diff_sklearn_vs_dequant.max():.6f}")
    print(f"Mean difference:     {diff_sklearn_vs_dequant.mean():.6f}")

    # Quality classification
    max_diff = diff_sklearn_vs_dequant.max()
    if max_diff < 0.1:
        print(" Quantization quality: Excellent")
    elif max_diff < 1.0:
        print(" Quantization quality: Acceptable")
    else:
        print(" Quantization quality: Poor")


def main():
    print(" Quantization pipeline started...\n")

    model = load_model("models/linear_regression_model.joblib")
    coef, intercept = save_params(model)
    quant_params = quantize_params(coef, intercept)
    dequant_coef, dequant_intercept = evaluate_dequantization(coef, intercept, quant_params)
    compare_predictions(model, coef, intercept, dequant_coef, dequant_intercept)

    print("\n Quantization process completed successfully.")


if __name__ == "__main__":
    main()
