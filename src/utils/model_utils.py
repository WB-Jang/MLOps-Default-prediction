"""Model management utilities."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch

from src.models import Encoder, ProjectionHead, TabTransformerClassifier


def save_model(
    model: torch.nn.Module,
    model_path: Path,
    model_version: str,
    metadata: Dict,
):
    """Save model to disk with metadata."""
    model_path.mkdir(parents=True, exist_ok=True)
    
    model_file = model_path / f"{model_version}.pt"
    metadata_file = model_path / f"{model_version}_metadata.json"
    
    # Save model state
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": metadata.get("model_config", {}),
            "timestamp": datetime.now().isoformat(),
        },
        model_file,
    )
    
    # Save metadata
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_file}")
    return str(model_file)


def load_model(
    model_path: Path,
    model_version: str,
    cnt_cat_features: int,
    cnt_num_features: int,
    cat_max_dict: Dict[int, int],
    device: str = "cpu",
) -> Tuple[TabTransformerClassifier, Dict]:
    """Load model from disk."""
    model_file = model_path / f"{model_version}.pt"
    metadata_file = model_path / f"{model_version}_metadata.json"
    
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    # Load checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    
    # Load metadata
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    
    # Reconstruct model
    model_config = checkpoint.get("model_config", {})
    
    encoder = Encoder(
        cnt_cat_features=cnt_cat_features,
        cnt_num_features=cnt_num_features,
        cat_max_dict=cat_max_dict,
        d_model=model_config.get("d_model", 32),
        nhead=model_config.get("nhead", 4),
        num_layers=model_config.get("num_layers", 6),
        dim_feedforward=model_config.get("dim_feedforward", 64),
        dropout_rate=model_config.get("dropout_rate", 0.3),
    )
    
    model = TabTransformerClassifier(
        encoder=encoder,
        d_model=model_config.get("d_model", 32),
        final_hidden=model_config.get("final_hidden", 128),
        dropout_rate=model_config.get("dropout_rate", 0.3),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, metadata


def get_latest_model_version(model_path: Path) -> str:
    """Get the latest model version from the model directory."""
    model_files = list(model_path.glob("*.pt"))
    if not model_files:
        return None
    
    # Sort by modification time
    latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
    return latest_file.stem
