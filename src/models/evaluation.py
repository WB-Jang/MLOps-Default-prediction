"""Model evaluation and fine-tuning utilities extracted from corp_default_modeling_f.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from loguru import logger
from typing import Dict, Tuple


def evaluate_model(model, test_loader, device='cpu') -> Dict[str, float]:
    """
    Evaluate model on test data.
    
    Args:
        model: The classifier model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        Dictionary with evaluation metrics (accuracy, roc_auc, f1_score)
    """
    model.to(device)
    model.eval()
    
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            cat_feats = batch["str"].to(device)
            num_feats = batch["num"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(cat_feats, num_feats)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs[:, 1].cpu().tolist())
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "f1_score": float(f1)
    }
    
    logger.info(f"Evaluation metrics - Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}, F1-score: {f1:.4f}")
    
    return metrics


def finetune_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device='cpu',
    epochs=3
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fine-tune model using the training loop from corp_default_modeling_f.py.
    
    Args:
        model: The classifier model to fine-tune
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data  
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epochs: Number of epochs
        
    Returns:
        Tuple of (training_metrics, test_metrics)
    """
    logger.info("Starting fine-tuning")
    model.to(device)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []
        
        for batch in train_loader:
            x_cat = batch["str"].to(device)
            x_num = batch["num"].to(device)
            y_true = batch["label"].to(device)
            
            logits = model(x_cat, x_num)
            loss = criterion(logits, y_true)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()
            all_labels.append(y_true.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())
        
        y_true_all = torch.cat(all_labels)
        y_prob_all = torch.cat(all_probs)
        y_pred_all = torch.cat(all_preds)
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(y_true_all, y_pred_all)
        epoch_auc = roc_auc_score(y_true_all, y_prob_all)
        epoch_f1 = f1_score(y_true_all, y_pred_all)
        
        logger.info(
            f"[Finetune] Epoch {epoch:2d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {epoch_acc:.4f} | "
            f"AUC: {epoch_auc:.4f} | "
            f"F1: {epoch_f1:.4f}"
        )
    
    # Final evaluation on test set
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            cat_feats = batch["str"].to(device)
            num_feats = batch["num"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(cat_feats, num_feats)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs[:, 1].cpu().tolist())
    
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    
    test_metrics = {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc),
        "f1_score": float(f1)
    }
    
    training_metrics = {
        "final_loss": float(epoch_loss),
        "final_accuracy": float(epoch_acc),
        "final_roc_auc": float(epoch_auc),
        "final_f1": float(epoch_f1)
    }
    
    logger.info(
        f"Final test metrics - "
        f"Accuracy: {acc:.4f}, "
        f"ROC-AUC: {roc_auc:.4f}, "
        f"F1-score: {f1:.4f}"
    )
    
    return training_metrics, test_metrics
