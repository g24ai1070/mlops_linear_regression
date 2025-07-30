import numpy as np
import os
from utils import load_model, load_dataset, calculate_metrics


def display_predictions(y_true, y_pred, num=10):
    print("\n Sample Predictions (first 10):")
    print("-" * 50)
    print(f"{'True':>10} | {'Predicted':>10} | {'Abs Diff':>10}")
    print("-" * 50)
    for i in range(min(num, len(y_true))):
        true_val = y_true[i]
        pred_val = y_pred[i]
        diff = abs(true_val - pred_val)
        print(f"{true_val:10.2f} | {pred_val:10.2f} | {diff:10.2f}")
    print("-" * 50)


def main():
    """Main prediction function for Docker container or local run."""

    model_path = "models/linear_regression_model.joblib"

    if not os.path.exists(model_path):
        print(f" Model file not found at: {model_path}")
        return False

    print(" Loading trained model...")
    model = load_model(model_path)

    print(" Loading test dataset...")
    _, X_test, _, y_test = load_dataset()

    print(" Making predictions...")
    y_pred = model.predict(X_test)

    print(" Calculating performance metrics...")
    r2, mse = calculate_metrics(y_test, y_pred)

    print("\n Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    display_predictions(y_test, y_pred)

    print("\n Prediction completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
