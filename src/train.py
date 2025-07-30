import numpy as np
from utils import (
    load_dataset,
    create_model,
    save_model,
    calculate_metrics
)


def main():
    """Main function for training the linear regression model."""
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("Creating model...")
    model = create_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    print("Calculating evaluation metrics...")
    r2_score, mse = calculate_metrics(y_test, y_pred)

    print(f"R² Score: {r2_score:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    print("Saving trained model...")
    model_path = "models/linear_regression_model.joblib"
    save_model(model, model_path)
    print(f"Model successfully saved to: {model_path}")

    return model, r2_score, mse


if __name__ == "__main__":
    main()
