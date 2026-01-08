"""Tests for MongoDB integration."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.database.mongodb import MongoDBClient


class TestMongoDBClient:
    """Test cases for MongoDB client."""
    
    @patch('src.database.mongodb.MongoClient')
    def test_connect(self, mock_mongo_client):
        """Test MongoDB connection."""
        client = MongoDBClient()
        client.connect()
        
        mock_mongo_client.assert_called_once()
        assert client.client is not None
        assert client.db is not None
    
    @patch('src.database.mongodb.MongoClient')
    def test_store_model_metadata(self, mock_mongo_client):
        """Test storing model metadata."""
        client = MongoDBClient()
        
        # Mock database and collection
        mock_collection = Mock()
        mock_collection.insert_one.return_value.inserted_id = "test_id_123"
        
        mock_db = Mock()
        mock_db.__getitem__.return_value = mock_collection
        
        client.db = mock_db
        
        model_id = client.store_model_metadata(
            model_name="test_model",
            model_path="/path/to/model.pth",
            model_version="v1.0",
            hyperparameters={"d_model": 32},
            metrics={"accuracy": 0.85}
        )
        
        assert model_id == "test_id_123"
        mock_collection.insert_one.assert_called_once()
    
    @patch('src.database.mongodb.MongoClient')
    def test_get_model_metadata(self, mock_mongo_client):
        """Test retrieving model metadata."""
        client = MongoDBClient()
        
        # Mock collection
        mock_collection = Mock()
        mock_collection.find_one.return_value = {
            "_id": "test_id",
            "model_name": "test_model",
            "model_path": "/path/to/model.pth",
            "model_version": "v1.0"
        }
        
        mock_db = Mock()
        mock_db.__getitem__.return_value = mock_collection
        
        client.db = mock_db
        
        metadata = client.get_model_metadata("test_model", "v1.0")
        
        assert metadata is not None
        assert metadata["model_name"] == "test_model"
        mock_collection.find_one.assert_called_once()
    
    @patch('src.database.mongodb.MongoClient')
    def test_store_prediction(self, mock_mongo_client):
        """Test storing predictions."""
        client = MongoDBClient()
        
        # Mock collection
        mock_collection = Mock()
        mock_collection.insert_one.return_value.inserted_id = "pred_id_123"
        
        mock_db = Mock()
        mock_db.__getitem__.return_value = mock_collection
        
        client.db = mock_db
        
        pred_id = client.store_prediction(
            model_id="model_123",
            input_data={"feature1": 1.0, "feature2": 2.0},
            prediction=0,
            confidence=0.95
        )
        
        assert pred_id == "pred_id_123"
        mock_collection.insert_one.assert_called_once()
    
    @patch('src.database.mongodb.MongoClient')
    def test_store_performance_metrics(self, mock_mongo_client):
        """Test storing performance metrics."""
        client = MongoDBClient()
        
        # Mock collection
        mock_collection = Mock()
        mock_collection.insert_one.return_value.inserted_id = "metrics_id_123"
        
        mock_db = Mock()
        mock_db.__getitem__.return_value = mock_collection
        
        client.db = mock_db
        
        metrics_id = client.store_performance_metrics(
            model_id="model_123",
            metrics={"accuracy": 0.85, "f1_score": 0.82},
            dataset_name="validation"
        )
        
        assert metrics_id == "metrics_id_123"
        mock_collection.insert_one.assert_called_once()
