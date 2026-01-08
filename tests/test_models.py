"""Tests for model components."""
import pytest
import torch

from src.models.network import (
    Encoder,
    TabTransformerClassifier,
    ProjectionHead,
    NTXentLoss
)


class TestEncoder:
    """Test cases for Encoder model."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        cat_max_dict = {i: 10 for i in range(5)}
        encoder = Encoder(
            cnt_cat_features=5,
            cnt_num_features=5,
            cat_max_dict=cat_max_dict,
            d_model=32,
            nhead=4,
            num_layers=2
        )
        
        assert encoder.d_model == 32
        assert encoder.cnt_cat_features == 5
        assert encoder.cnt_num_features == 5
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        cat_max_dict = {i: 10 for i in range(5)}
        encoder = Encoder(
            cnt_cat_features=5,
            cnt_num_features=5,
            cat_max_dict=cat_max_dict,
            d_model=32,
            nhead=4,
            num_layers=2
        )
        
        batch_size = 4
        x_cat = torch.randint(0, 10, (batch_size, 5))
        x_num = torch.randn(batch_size, 5)
        
        output = encoder(x_cat, x_num)
        
        assert output.shape == (batch_size, 10, 32)  # (batch, seq_len, d_model)


class TestTabTransformerClassifier:
    """Test cases for TabTransformerClassifier."""
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        cat_max_dict = {i: 10 for i in range(5)}
        encoder = Encoder(
            cnt_cat_features=5,
            cnt_num_features=5,
            cat_max_dict=cat_max_dict,
            d_model=32
        )
        
        classifier = TabTransformerClassifier(
            encoder=encoder,
            d_model=32,
            final_hidden=128
        )
        
        assert classifier.encoder is not None
    
    def test_classifier_forward(self):
        """Test classifier forward pass."""
        cat_max_dict = {i: 10 for i in range(5)}
        encoder = Encoder(
            cnt_cat_features=5,
            cnt_num_features=5,
            cat_max_dict=cat_max_dict,
            d_model=32
        )
        
        classifier = TabTransformerClassifier(encoder=encoder, d_model=32)
        
        batch_size = 4
        x_cat = torch.randint(0, 10, (batch_size, 5))
        x_num = torch.randn(batch_size, 5)
        
        output = classifier(x_cat, x_num)
        
        assert output.shape == (batch_size, 2)  # Binary classification


class TestProjectionHead:
    """Test cases for ProjectionHead."""
    
    def test_projection_head_forward(self):
        """Test projection head forward pass."""
        head = ProjectionHead(d_model=32, projection_dim=128)
        
        batch_size = 4
        x = torch.randn(batch_size, 32)
        
        output = head(x)
        
        assert output.shape == (batch_size, 128)


class TestNTXentLoss:
    """Test cases for NT-Xent loss."""
    
    def test_ntxent_loss_calculation(self):
        """Test NT-Xent loss calculation."""
        loss_fn = NTXentLoss(temperature=0.5)
        
        batch_size = 4
        projection_dim = 128
        
        z_i = torch.randn(batch_size, projection_dim)
        z_j = torch.randn(batch_size, projection_dim)
        
        loss = loss_fn(z_i, z_j)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestModelSaving:
    """Test cases for model saving in .pth format."""
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading model in .pth format."""
        cat_max_dict = {i: 10 for i in range(5)}
        encoder = Encoder(
            cnt_cat_features=5,
            cnt_num_features=5,
            cat_max_dict=cat_max_dict,
            d_model=32
        )
        
        classifier = TabTransformerClassifier(encoder=encoder, d_model=32)
        
        # Save model
        save_path = tmp_path / "test_model.pth"
        torch.save({
            'model_state_dict': classifier.state_dict(),
        }, save_path)
        
        assert save_path.exists()
        assert save_path.suffix == '.pth'
        
        # Load model
        checkpoint = torch.load(save_path)
        assert 'model_state_dict' in checkpoint
