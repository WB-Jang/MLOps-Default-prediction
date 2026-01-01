"""Test configuration and database utilities."""
import pytest


def test_settings_import():
    """Test that settings can be imported."""
    from config.settings import settings
    
    assert settings is not None
    assert settings.f1_score_threshold == 0.8
    assert settings.d_model == 32


def test_database_manager_init():
    """Test database manager initialization."""
    from src.data import DatabaseManager
    
    # Test with dummy URL
    db = DatabaseManager("postgresql://user:pass@localhost/test")
    assert db is not None
    assert db.database_url == "postgresql://user:pass@localhost/test"


def test_model_imports():
    """Test that model classes can be imported."""
    from src.models import (
        Encoder,
        NTXentLoss,
        ProjectionHead,
        TabTransformerClassifier,
        tabular_augment,
    )
    
    assert Encoder is not None
    assert TabTransformerClassifier is not None
    assert ProjectionHead is not None
    assert NTXentLoss is not None
    assert tabular_augment is not None
