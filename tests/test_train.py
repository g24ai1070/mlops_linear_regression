import os
import sys
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

# Add src/ directory to path for module import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    load_dataset,
    create_model,
    save_model,
    load_model,
    calculate_metrics
)


class TestMLPipeline:
    """Unit tests for the California Housing Price Prediction ML pipeline."""

    def test_load_dataset(self):
        """Ensure the dataset loads correctly and is properly split."""
        X_train, X_test, y_train, y_test = load_dataset()

        assert X_train.shape[1] == 8
        assert X_test.shape[1] == 8
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # Roughly 80/20 split
        ratio = len(X_train) / (len(X_train) + len(X_test))
        assert 0.75 <= ratio <= 0.85

    def test_create_model(self):
        """Ensure a LinearRegression model is created properly."""
        model = create_model()
        assert isinstance(model, LinearRegression)
        assert hasattr(model, 'fit') and hasattr(model, 'predict')

    def test_train_model_attributes(self):
        """Train model and verify key learned attributes."""
        X_train, _, y_train, _ = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)

        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_.shape == (8,)
        assert isinstance(model.intercept_, (float, np.floating))

    def test_model_performance(self):
        """Verify model accuracy using R² and MSE thresholds."""
        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2, mse = calculate_metrics(y_test, y_pred)

        assert r2 >= 0.5, f"R² too low: {r2:.4f}"
        assert mse > 0, "MSE should be a positive value"

        print(f"R² Score: {r2:.4f}")
        print(f"Mean Squared Error: {mse:.4f}")

    def test_model_save_and_load(self):
        """Ensure the model can be saved and loaded with consistent output."""
        X_train, X_test, y_train, _ = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)

        filename = "test_model.joblib"
        save_model(model, filename)
        assert os.path.exists(filename)

        loaded_model = load_model(filename)
        original_pred = model.predict(X_test[:5])
        loaded_pred = loaded_model.predict(X_test[:5])

        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)

        os.remove(filename)  # Clean up


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
