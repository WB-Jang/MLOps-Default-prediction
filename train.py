"""Standalone training script for model training."""
import os
import argparse
import torch
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.models import (
    Encoder,
    TabTransformerClassifier,
    ProjectionHead,
    pretrain_contrastive,
    train_classifier
)
from src.database.mongodb import mongodb_client
from config.settings import settings


def setup_logging():
    """Setup logging configuration."""
    logger.add(
        "logs/training_{time}.log",
        rotation="500 MB",
        level=settings.log_level
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train loan default prediction model')
    parser.add_argument('--pretrain', action='store_true', help='Run pretraining')
    parser.add_argument('--train', action='store_true', help='Train classifier')
    parser.add_argument('--epochs', type=int, default=settings.epochs, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=settings.batch_size, help='Batch size')
    parser.add_argument('--lr', type=float, default=settings.learning_rate, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default=settings.model_save_path, help='Model save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training configuration: {args}")
    logger.info(f"Using device: {args.device}")
    
    # In production, load actual training data here
    # For demonstration, we'll initialize models with placeholder parameters
    
    # Placeholder categorical max dictionary
    cat_max_dict = {i: 100 for i in range(10)}
    
    # Create encoder
    encoder = Encoder(
        cnt_cat_features=10,
        cnt_num_features=10,
        cat_max_dict=cat_max_dict,
        d_model=settings.d_model,
        nhead=settings.nhead,
        num_layers=settings.num_layers,
        dim_feedforward=settings.dim_feedforward,
        dropout_rate=settings.dropout_rate
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Pretraining
    if args.pretrain:
        logger.info("Starting pretraining...")
        projection_head = ProjectionHead(
            d_model=settings.d_model,
            projection_dim=128
        )
        
        # In production, pass actual data loader
        # pretrain_contrastive(encoder, projection_head, train_loader, ...)
        
        # Save pretrained model
        pretrain_path = save_dir / f"pretrained_encoder_{timestamp}.pth"
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'projection_head_state_dict': projection_head.state_dict(),
            'hyperparameters': {
                'd_model': settings.d_model,
                'nhead': settings.nhead,
                'num_layers': settings.num_layers,
                'dim_feedforward': settings.dim_feedforward,
                'dropout_rate': settings.dropout_rate
            }
        }, pretrain_path)
        
        logger.info(f"Pretrained encoder saved to {pretrain_path}")
        
        # Store in MongoDB
        mongodb_client.connect()
        try:
            model_id = mongodb_client.store_model_metadata(
                model_name="default_prediction_encoder",
                model_path=str(pretrain_path),
                model_version=timestamp,
                hyperparameters={
                    "d_model": settings.d_model,
                    "nhead": settings.nhead,
                    "num_layers": settings.num_layers,
                    "dim_feedforward": settings.dim_feedforward,
                    "dropout_rate": settings.dropout_rate
                }
            )
            logger.info(f"Model metadata stored in MongoDB with ID: {model_id}")
        finally:
            mongodb_client.disconnect()
    
    # Classifier training
    if args.train:
        logger.info("Starting classifier training...")
        
        classifier = TabTransformerClassifier(
            encoder=encoder,
            d_model=settings.d_model,
            final_hidden=128,
            dropout_rate=settings.dropout_rate
        )
        
        # In production, pass actual data loaders
        # metrics = train_classifier(classifier, train_loader, val_loader, ...)
        
        # Save trained classifier
        classifier_path = save_dir / f"classifier_{timestamp}.pth"
        torch.save({
            'model_state_dict': classifier.state_dict(),
            'hyperparameters': {
                'd_model': settings.d_model,
                'final_hidden': 128,
                'dropout_rate': settings.dropout_rate
            }
        }, classifier_path)
        
        logger.info(f"Classifier saved to {classifier_path}")
        
        # Store in MongoDB
        mongodb_client.connect()
        try:
            model_id = mongodb_client.store_model_metadata(
                model_name="default_prediction_classifier",
                model_path=str(classifier_path),
                model_version=timestamp,
                hyperparameters={
                    "d_model": settings.d_model,
                    "final_hidden": 128,
                    "dropout_rate": settings.dropout_rate
                },
                metrics={}  # Add actual metrics from training
            )
            logger.info(f"Classifier metadata stored in MongoDB with ID: {model_id}")
        finally:
            mongodb_client.disconnect()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
