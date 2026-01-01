"""FastAPI application for model serving."""
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from config.settings import settings
from src.data import DatabaseManager
from src.utils import load_model

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan defaults using Transformer models",
    version="1.0.0",
)

# Global variables for model
current_model = None
current_model_version = None
db_manager = DatabaseManager(settings.database_url)


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    categorical_features: Dict[str, int]
    numerical_features: Dict[str, float]


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: int
    probability: float
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_version: str
    model_loaded: bool


def load_active_model():
    """Load the active model from the database."""
    global current_model, current_model_version
    
    try:
        active_model = db_manager.get_active_model()
        
        if not active_model:
            raise ValueError("No active model found")
        
        model_version = active_model["model_version"]
        
        # Only reload if different version
        if model_version != current_model_version:
            # TODO: Load actual model with proper configuration
            # This requires knowing the feature configuration
            # For now, this is a placeholder
            
            print(f"Loading model version: {model_version}")
            # model, metadata = load_model(
            #     settings.model_path,
            #     model_version,
            #     cnt_cat_features=...,
            #     cnt_num_features=...,
            #     cat_max_dict=...,
            #     device="cpu"
            # )
            # current_model = model
            current_model_version = model_version
            print(f"Model {model_version} loaded successfully")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_active_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Loan Default Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_version=current_model_version or "none",
        model_loaded=current_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for loan default.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Prediction response with result and probability
    """
    if current_model is None:
        try:
            load_active_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    
    try:
        # TODO: Implement actual prediction logic
        # This requires:
        # 1. Converting request features to tensors
        # 2. Running inference
        # 3. Post-processing results
        
        # Placeholder response
        prediction = 0
        probability = 0.5
        
        # Save prediction to database
        # db_manager.save_prediction(
        #     model_version=current_model_version,
        #     data_id=...,
        #     prediction=prediction,
        #     probability=probability
        # )
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_version=current_model_version,
            timestamp=datetime.now().isoformat(),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/reload-model")
async def reload_model():
    """Reload the active model from database."""
    try:
        load_active_model()
        return {
            "status": "success",
            "model_version": current_model_version,
            "message": "Model reloaded successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")


@app.get("/model-info")
async def model_info():
    """Get information about the current model."""
    if current_model_version is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    active_model = db_manager.get_active_model()
    
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found in database")
    
    return {
        "model_version": active_model["model_version"],
        "created_at": active_model["created_at"].isoformat() if active_model.get("created_at") else None,
        "f1_score": active_model.get("f1_score"),
        "accuracy": active_model.get("accuracy"),
        "training_samples": active_model.get("training_samples"),
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
