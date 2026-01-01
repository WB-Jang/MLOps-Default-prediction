"""Training utilities for the loan default prediction model."""
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models import NTXentLoss, tabular_augment


def pretrain_contrastive(encoder, projection_head, loader, epochs=10, device="cpu", lr=3e-4):
    """
    Pretrain encoder using contrastive learning.

    Args:
        encoder: Encoder model
        projection_head: Projection head model
        loader: DataLoader for pretraining data
        epochs: Number of epochs
        device: Device to train on
        lr: Learning rate

    Returns:
        Trained encoder and projection head
    """
    print("--- Starting Contrastive Learning Pre-training ---")
    encoder.to(device).train()
    projection_head.to(device).train()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()), lr=lr
    )
    criterion = NTXentLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x_cat, x_num = batch["cat"].to(device), batch["num"].to(device)

            # Generate two augmented views
            x_cat_i, x_num_i = tabular_augment(x_cat, x_num)
            x_cat_j, x_num_j = tabular_augment(x_cat, x_num)

            # Encode both views
            h_i = encoder(x_cat_i, x_num_i).mean(dim=1)  # (B, d_model)
            h_j = encoder(x_cat_j, x_num_j).mean(dim=1)  # (B, d_model)

            # Project to contrastive space
            z_i = projection_head(h_i)  # (B, projection_dim)
            z_j = projection_head(h_j)  # (B, projection_dim)

            # Compute NT-Xent loss
            loss = criterion(z_i, z_j)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Pretrain] Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    return encoder, projection_head


def train_classifier(
    model, train_loader, val_loader, epochs=50, device="cpu", lr=3e-4, patience=5
):
    """
    Train the classifier model.

    Args:
        model: TabTransformerClassifier model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        device: Device to train on
        lr: Learning rate
        patience: Early stopping patience

    Returns:
        Trained model and best metrics
    """
    print("--- Starting Classifier Training ---")
    model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_f1 = 0.0
    best_metrics = {}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_cat, x_num, y = (
                batch["cat"].to(device),
                batch["num"].to(device),
                batch["target"].to(device),
            )

            optimizer.zero_grad()
            outputs = model(x_cat, x_num)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)

        print(
            f"[Train] Epoch {epoch+1}, "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Val F1: {val_metrics['f1_score']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Learning rate scheduling
        scheduler.step(val_metrics["f1_score"])

        # Early stopping and model saving
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    return model, best_metrics


def evaluate_model(model, loader, device="cpu"):
    """
    Evaluate model performance.

    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to evaluate on

    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_cat, x_num, y = (
                batch["cat"].to(device),
                batch["num"].to(device),
                batch["target"].to(device),
            )

            outputs = model(x_cat, x_num)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_targets, all_preds),
        "f1_score": f1_score(all_targets, all_preds, average="binary"),
        "precision": precision_score(all_targets, all_preds, average="binary"),
        "recall": recall_score(all_targets, all_preds, average="binary"),
        "roc_auc": roc_auc_score(all_targets, all_probs),
    }

    return metrics
