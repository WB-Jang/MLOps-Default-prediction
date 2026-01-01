"""Configuration management for the MLOps pipeline."""
import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Database
    database_url: str = "postgresql://mlops_user:mlops_password@localhost:5432/loan_default"
    airflow_db_url: str = (
        "postgresql+psycopg2://mlops_user:mlops_password@localhost:5432/airflow_db"
    )

    # Model paths
    model_path: Path = Path("./models")
    f1_score_threshold: float = 0.8
    retraining_sample_threshold: int = 1000

    # Model hyperparameters
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 6
    dim_feedforward: int = 64
    dropout_rate: float = 0.3
    final_hidden: int = 128
    projection_dim: int = 128

    # Training configuration
    batch_size: int = 32
    learning_rate: float = 0.0003
    num_epochs: int = 50
    early_stopping_patience: int = 5
    temperature: float = 0.5
    mask_ratio: float = 0.15
    pretrain_epochs: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
