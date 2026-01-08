"""Test configuration."""
import pytest


@pytest.fixture
def sample_categorical_data():
    """Sample categorical data for testing."""
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


@pytest.fixture
def sample_numerical_data():
    """Sample numerical data for testing."""
    return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


@pytest.fixture
def sample_model_metadata():
    """Sample model metadata for testing."""
    return {
        "model_name": "test_model",
        "model_path": "/tmp/test_model.pth",
        "model_version": "v1.0.0",
        "hyperparameters": {
            "d_model": 32,
            "nhead": 4,
            "num_layers": 6
        }
    }
