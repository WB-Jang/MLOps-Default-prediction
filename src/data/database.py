"""Database utilities for data management."""
import json
from datetime import date, datetime
from typing import Dict, List, Optional

import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text


class DatabaseManager:
    """Manager for database operations."""

    def __init__(self, database_url: str):
        """Initialize database manager."""
        self.database_url = database_url
        self.engine = create_engine(database_url)

    def insert_raw_data(self, data_date: date, cat_features: dict, num_features: dict, target: int):
        """Insert raw data into the database."""
        with self.engine.connect() as conn:
            query = text(
                """
                INSERT INTO loan_data.raw_data 
                (data_date, categorical_features, numerical_features, target)
                VALUES (:data_date, :cat_features, :num_features, :target)
                ON CONFLICT DO NOTHING
                RETURNING id
                """
            )
            result = conn.execute(
                query,
                {
                    "data_date": data_date,
                    "cat_features": json.dumps(cat_features),
                    "num_features": json.dumps(num_features),
                    "target": target,
                },
            )
            conn.commit()
            return result.fetchone()

    def get_new_data_for_training(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Retrieve new data that hasn't been used for training yet."""
        query = """
            SELECT 
                id,
                categorical_features,
                numerical_features,
                target
            FROM loan_data.raw_data
            WHERE id NOT IN (
                SELECT DISTINCT raw_data_id 
                FROM loan_data.processed_data
            )
            ORDER BY created_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"

        return pd.read_sql(query, self.engine)

    def save_model_metadata(
        self,
        model_version: str,
        model_path: str,
        training_samples: int,
        metrics: Dict[str, float],
        training_config: Dict,
        is_active: bool = False,
    ):
        """Save model metadata to the database."""
        with self.engine.connect() as conn:
            query = text(
                """
                INSERT INTO loan_data.model_metadata
                (model_version, model_path, training_samples, f1_score, 
                 accuracy, precision_score, recall, roc_auc, is_active, training_config)
                VALUES (:model_version, :model_path, :training_samples, :f1_score,
                        :accuracy, :precision_score, :recall, :roc_auc, :is_active, :training_config)
                ON CONFLICT (model_version) DO UPDATE SET
                    f1_score = EXCLUDED.f1_score,
                    accuracy = EXCLUDED.accuracy,
                    precision_score = EXCLUDED.precision_score,
                    recall = EXCLUDED.recall,
                    roc_auc = EXCLUDED.roc_auc,
                    is_active = EXCLUDED.is_active
                """
            )
            conn.execute(
                query,
                {
                    "model_version": model_version,
                    "model_path": model_path,
                    "training_samples": training_samples,
                    "f1_score": metrics.get("f1_score"),
                    "accuracy": metrics.get("accuracy"),
                    "precision_score": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "roc_auc": metrics.get("roc_auc"),
                    "is_active": is_active,
                    "training_config": json.dumps(training_config),
                },
            )
            conn.commit()

    def get_active_model(self) -> Optional[Dict]:
        """Get the currently active model metadata."""
        query = """
            SELECT * FROM loan_data.model_metadata
            WHERE is_active = true
            ORDER BY created_at DESC
            LIMIT 1
        """
        result = pd.read_sql(query, self.engine)
        if len(result) > 0:
            return result.iloc[0].to_dict()
        return None

    def set_active_model(self, model_version: str):
        """Set a model as active and deactivate others."""
        with self.engine.connect() as conn:
            # Deactivate all models
            conn.execute(
                text("UPDATE loan_data.model_metadata SET is_active = false WHERE is_active = true")
            )
            # Activate the specified model
            conn.execute(
                text(
                    "UPDATE loan_data.model_metadata SET is_active = true WHERE model_version = :version"
                ),
                {"version": model_version},
            )
            conn.commit()

    def save_prediction(
        self, model_version: str, data_id: int, prediction: int, probability: float
    ):
        """Save a prediction to the database."""
        with self.engine.connect() as conn:
            query = text(
                """
                INSERT INTO loan_data.predictions
                (model_version, data_id, prediction, probability)
                VALUES (:model_version, :data_id, :prediction, :probability)
                """
            )
            conn.execute(
                query,
                {
                    "model_version": model_version,
                    "data_id": data_id,
                    "prediction": prediction,
                    "probability": probability,
                },
            )
            conn.commit()

    def save_model_performance(
        self, model_version: str, evaluation_date: date, metrics: Dict[str, float], sample_count: int
    ):
        """Save model performance metrics."""
        with self.engine.connect() as conn:
            query = text(
                """
                INSERT INTO loan_data.model_performance
                (model_version, evaluation_date, f1_score, accuracy, 
                 precision_score, recall, roc_auc, sample_count)
                VALUES (:model_version, :evaluation_date, :f1_score, :accuracy,
                        :precision_score, :recall, :roc_auc, :sample_count)
                """
            )
            conn.execute(
                query,
                {
                    "model_version": model_version,
                    "evaluation_date": evaluation_date,
                    "f1_score": metrics.get("f1_score"),
                    "accuracy": metrics.get("accuracy"),
                    "precision_score": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "roc_auc": metrics.get("roc_auc"),
                    "sample_count": sample_count,
                },
            )
            conn.commit()

    def get_recent_performance(self, model_version: str, days: int = 30) -> pd.DataFrame:
        """Get recent performance metrics for a model."""
        query = """
            SELECT * FROM loan_data.model_performance
            WHERE model_version = :model_version
            AND evaluation_date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY evaluation_date DESC
        """
        return pd.read_sql(query, self.engine, params={"model_version": model_version})

    def check_retraining_needed(self, threshold: float = 0.8) -> bool:
        """Check if retraining is needed based on recent performance."""
        active_model = self.get_active_model()
        if not active_model:
            return True

        query = """
            SELECT AVG(f1_score) as avg_f1
            FROM loan_data.model_performance
            WHERE model_version = :model_version
            AND evaluation_date >= CURRENT_DATE - INTERVAL '7 days'
        """
        result = pd.read_sql(query, self.engine, params={"model_version": active_model["model_version"]})

        if len(result) > 0 and result.iloc[0]["avg_f1"] is not None:
            return result.iloc[0]["avg_f1"] < threshold
        return False
