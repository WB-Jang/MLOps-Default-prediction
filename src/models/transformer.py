"""Transformer-based model architecture for loan default prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Transformer Encoder for tabular data."""

    def __init__(
        self,
        cnt_cat_features,
        cnt_num_features,
        cat_max_dict,
        d_model=32,
        nhead=4,
        num_layers=6,
        dim_feedforward=64,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.cnt_cat_features = cnt_cat_features
        self.cnt_num_features = cnt_num_features

        # Categorical feature embeddings
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cat_max_dict[i], d_model) for i in range(cnt_cat_features)]
        )

        # Numerical feature embeddings
        self.num_embeddings = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(cnt_num_features)]
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_cat, x_num):
        """Forward pass through the encoder."""
        # Embed categorical features
        cat_emb = torch.cat(
            [self.embeddings[i](x_cat[:, i]).unsqueeze(1) for i in range(self.cnt_cat_features)],
            dim=1,
        )

        # Embed numerical features
        num_emb = torch.cat(
            [
                self.num_embeddings[i](x_num[:, i].unsqueeze(1)).unsqueeze(1)
                for i in range(self.cnt_num_features)
            ],
            dim=1,
        )

        # Concatenate embeddings
        x = torch.cat([cat_emb, num_emb], dim=1)

        # Pass through transformer
        x = self.transformer(x)
        return x


class TabTransformerClassifier(nn.Module):
    """Classification model with pretrained encoder."""

    def __init__(self, encoder, d_model=32, final_hidden=128, dropout_rate=0.3):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(d_model, final_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_hidden, 2),
        )

    def forward(self, x_cat, x_num):
        """Forward pass through the classifier."""
        encoded = self.encoder(x_cat, x_num)
        # Pool across all features to get a single representation
        pooled = encoded.mean(dim=1)
        return self.fc(pooled)


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning."""

    def __init__(self, d_model=32, projection_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, projection_dim)
        )

    def forward(self, x):
        """Forward pass through projection head."""
        return self.head(x)


class NTXentLoss(nn.Module):
    """NT-Xent loss for contrastive learning."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss."""
        B, _ = z_i.shape
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, B)
        sim_j_i = torch.diag(sim, -B)

        # Positive pairs
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(2 * B, 1)

        # Negative pairs (excluding self-similarity)
        mask = torch.ones((2 * B, 2 * B), dtype=bool, device=z.device)
        mask.fill_diagonal_(False)
        negative_samples = sim[mask].reshape(2 * B, -1)

        # Final logits
        logits = torch.cat([positive_samples, negative_samples], dim=1)

        # Labels: first position is always the positive pair
        labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)

        return self.criterion(logits, labels)
