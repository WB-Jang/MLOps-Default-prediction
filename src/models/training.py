"""Training utilities for model training and pretraining."""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from loguru import logger

from .network import Encoder, TabTransformerClassifier, ProjectionHead, NTXentLoss


def tabular_augment(
    x_cat: torch.Tensor,
    x_num: torch.Tensor,
    mask_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Augment tabular data by randomly masking features.
    
    Args:
        x_cat: Categorical features tensor
        x_num: Numerical features tensor
        mask_ratio: Ratio of features to mask
        
    Returns:
        Tuple of augmented categorical and numerical tensors
    """
    # Augment categorical features
    cat_mask = torch.rand(x_cat.shape, device=x_cat.device) < mask_ratio
    x_cat_aug = x_cat.clone()
    x_cat_aug[cat_mask] = 0
    
    # Augment numerical features
    num_mask = torch.rand(x_num.shape, device=x_num.device) < mask_ratio
    x_num_aug = x_num.clone()
    x_num_aug[num_mask] = 0.0
    
    return x_cat_aug, x_num_aug


def pretrain_contrastive(
    encoder: Encoder,
    projection_head: ProjectionHead,
    loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    lr: float = 3e-4,
    temperature: float = 0.5,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> None:
    """
    Pretrain encoder using contrastive learning.
    
    Args:
        encoder: Encoder model to pretrain
        projection_head: Projection head for contrastive learning
        loader: Data loader for pretraining
        epochs: Number of pretraining epochs
        lr: Learning rate
        temperature: Temperature for NT-Xent loss
        device: Device to train on
        save_path: Path to save pretrained encoder (in .pth format)
    """
    logger.info("Starting contrastive learning pretraining")
    encoder.to(device).train()
    projection_head.to(device).train()
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=lr
    )
    criterion = NTXentLoss(temperature=temperature)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x_cat = batch["str"].to(device)
            x_num = batch["num"].to(device)
            
            # Generate two augmented views
            x_cat_i, x_num_i = tabular_augment(x_cat, x_num)
            x_cat_j, x_num_j = tabular_augment(x_cat, x_num)
            
            # Encode and project both views
            h_i = encoder(x_cat_i, x_num_i).mean(dim=1)
            h_j = encoder(x_cat_j, x_num_j).mean(dim=1)
            
            z_i = projection_head(h_i)
            z_j = projection_head(h_j)
            
            # Calculate NT-Xent loss
            loss = criterion(z_i, z_j)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        logger.info(f"[Pretrain] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save pretrained encoder in .pth format
    if save_path:
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'projection_head_state_dict': projection_head.state_dict(),
        }, save_path)
        logger.info(f"Pretrained encoder saved to {save_path}")


def train_classifier(
    model: TabTransformerClassifier,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 10,
    lr: float = 3e-4,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Train the classifier model.
    
    Args:
        model: TabTransformerClassifier to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save trained model (in .pth format)
        
    Returns:
        Dictionary with training metrics
    """
    logger.info("Starting classifier training")
    model.to(device).train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    metrics = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            x_cat = batch["str"].to(device)
            x_num = batch["num"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            logits = model(x_cat, x_num)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        metrics['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if val_loader:
            model.eval()
            total_val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    x_cat = batch["str"].to(device)
                    x_num = batch["num"].to(device)
                    labels = batch["label"].to(device)
                    
                    logits = model(x_cat, x_num)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            metrics['val_loss'].append(avg_val_loss)
            
            logger.info(
                f"[Train] Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )
            
            # Save best model
            if save_path and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                }, save_path)
                logger.info(f"Best model saved to {save_path}")
        else:
            logger.info(f"[Train] Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
    
    # Save final model if no validation set
    if save_path and not val_loader:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        logger.info(f"Final model saved to {save_path}")
    
    return metrics


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load model from .pth checkpoint.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to .pth checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logger.info(f"Model loaded from {checkpoint_path}")
    return model
