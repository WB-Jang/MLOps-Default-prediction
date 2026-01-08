"""MongoDB database connection and operations."""
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from bson import ObjectId
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

from config.settings import settings


class MongoDBClient:
    """MongoDB client for managing connections and operations."""
    
    def __init__(self):
        """Initialize MongoDB client."""
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        
    def connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(settings.mongodb_connection_string)
            self.db = self.client[settings.mongodb_database]
            # Test connection
            self.client.server_info()
            logger.info(f"Connected to MongoDB: {settings.mongodb_database}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_collection(self, collection_name: str) -> Collection:
        """
        Get a collection from the database.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB collection object
        """
        if not self.db:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db[collection_name]
    
    def store_model_metadata(
        self,
        model_name: str,
        model_path: str,
        model_version: str,
        hyperparameters: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Store model metadata in MongoDB.
        
        Args:
            model_name: Name of the model
            model_path: Path to saved model file (.pth)
            model_version: Version of the model
            hyperparameters: Model hyperparameters
            metrics: Training/validation metrics
            
        Returns:
            Document ID
        """
        collection = self.get_collection("model_metadata")
        
        document = {
            "model_name": model_name,
            "model_path": model_path,
            "model_version": model_version,
            "model_format": "pth",
            "hyperparameters": hyperparameters,
            "metrics": metrics or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = collection.insert_one(document)
        logger.info(f"Model metadata stored with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def get_model_metadata(
        self,
        model_name: str,
        model_version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve model metadata from MongoDB.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model (optional, gets latest if not specified)
            
        Returns:
            Model metadata document or None
        """
        collection = self.get_collection("model_metadata")
        
        query = {"model_name": model_name}
        if model_version:
            query["model_version"] = model_version
        
        if model_version:
            return collection.find_one(query)
        else:
            # Get latest version
            return collection.find_one(query, sort=[("created_at", -1)])
    
    def store_performance_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float],
        dataset_name: str = "validation"
    ) -> str:
        """
        Store model performance metrics.
        
        Args:
            model_id: ID of the model
            metrics: Performance metrics dictionary
            dataset_name: Name of the dataset used for evaluation
            
        Returns:
            Document ID
        """
        collection = self.get_collection("performance_metrics")
        
        document = {
            "model_id": model_id,
            "dataset_name": dataset_name,
            "metrics": metrics,
            "timestamp": datetime.utcnow()
        }
        
        result = collection.insert_one(document)
        logger.info(f"Performance metrics stored with ID: {result.inserted_id}")
        return str(result.inserted_id)
    
    def store_prediction(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        prediction: Any,
        confidence: Optional[float] = None
    ) -> str:
        """
        Log a prediction to MongoDB.
        
        Args:
            model_id: ID of the model that made the prediction
            input_data: Input features
            prediction: Model prediction
            confidence: Prediction confidence score
            
        Returns:
            Document ID
        """
        collection = self.get_collection("predictions")
        
        document = {
            "model_id": model_id,
            "input_data": input_data,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.utcnow()
        }
        
        result = collection.insert_one(document)
        return str(result.inserted_id)
    
    def get_predictions(
        self,
        model_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve predictions from MongoDB.
        
        Args:
            model_id: Filter by model ID (optional)
            limit: Maximum number of predictions to retrieve
            
        Returns:
            List of prediction documents
        """
        collection = self.get_collection("predictions")
        
        query = {}
        if model_id:
            query["model_id"] = model_id
        
        cursor = collection.find(query).sort("timestamp", -1).limit(limit)
        return list(cursor)
    
    def update_model_status(
        self,
        model_id: str,
        status: str,
        notes: Optional[str] = None
    ) -> None:
        """
        Update model status in MongoDB.
        
        Args:
            model_id: ID of the model (can be string or ObjectId)
            status: New status (e.g., 'training', 'deployed', 'archived')
            notes: Optional notes about the status change
        """
        collection = self.get_collection("model_metadata")
        
        # Convert string ID to ObjectId if needed
        try:
            object_id = ObjectId(model_id) if isinstance(model_id, str) else model_id
        except Exception:
            # If not a valid ObjectId, use as is (custom string ID)
            object_id = model_id
        
        update_doc = {
            "$set": {
                "status": status,
                "updated_at": datetime.utcnow()
            }
        }
        
        if notes:
            update_doc["$set"]["notes"] = notes
        
        collection.update_one({"_id": object_id}, update_doc)
        logger.info(f"Model {model_id} status updated to: {status}")


# Global MongoDB client instance
mongodb_client = MongoDBClient()
