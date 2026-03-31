"""Autoencoder model for chemical abundance encoding and decoding."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets import DatasetConfig
from src.models.autoencoder.config import AEConfig


class Autoencoder(nn.Module):
    """Autoencoder neural network for encoding/decoding chemical abundances."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 12,
        hidden_dims: tuple[int, ...] = (320, 160),
        noise: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize autoencoder with encoder and decoder layers."""
        super().__init__()

        self.encoder_fc1 = nn.Linear(input_dim, hidden_dims[0], bias=False)
        self.encoder_bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.encoder_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1], bias=False)
        self.encoder_bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.encoder_fc3 = nn.Linear(hidden_dims[1], latent_dim)
        self.encoder_norm3 = nn.BatchNorm1d(latent_dim)

        self.decoder_bn1 = nn.BatchNorm1d(hidden_dims[1])
        self.decoder_bn2 = nn.BatchNorm1d(hidden_dims[0])

        self.decoder_bias1 = nn.Parameter(torch.zeros(hidden_dims[1]))
        self.decoder_bias2 = nn.Parameter(torch.zeros(hidden_dims[0]))
        self.decoder_bias3 = nn.Parameter(torch.zeros(input_dim))

        self.activation = nn.GELU()
        self.final_activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.noise = noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to latent space representation."""
        x = self.activation(self.encoder_bn1(self.encoder_fc1(x)))
        x = self.activation(self.encoder_bn2(self.encoder_fc2(x)))
        x = self.dropout(x)
        return self.activation(self.encoder_norm3(self.encoder_fc3(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent space representation back to input space."""
        z = F.linear(z, self.encoder_fc3.weight.t()) + self.decoder_bias1
        z = self.activation(self.decoder_bn1(z))

        z = F.linear(z, self.encoder_fc2.weight.t()) + self.decoder_bias2
        z = self.activation(self.decoder_bn2(z))
        z = self.dropout(z)

        x_reconstructed = F.linear(z, self.encoder_fc1.weight.t()) + self.decoder_bias3
        return self.final_activation(x_reconstructed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder with optional noise."""
        z = self.encode(x)
        if self.training and self.noise > 0:
            z = z + torch.randn_like(z) * self.noise
        return self.decode(z)


def load_autoencoder(
    autoencoder_class: type[Autoencoder],
    general_config: DatasetConfig,
    ae_config: AEConfig,
    inference: bool = False,
) -> Autoencoder:
    """Load autoencoder model with optional pretrained weights."""
    autoencoder = autoencoder_class(
        input_dim=ae_config.input_dim,
        latent_dim=ae_config.latent_dim,
        hidden_dims=ae_config.hidden_dims,
        noise=ae_config.noise,
        dropout=ae_config.dropout,
    ).to(general_config.device)
    if os.path.exists(ae_config.pretrained_model_path):
        print("Loading Pretrained Model")
        autoencoder.load_state_dict(
            torch.load(
                ae_config.pretrained_model_path,
                map_location=torch.device("cpu"),
            )
        )

    if inference:
        print("Setting Autoencoder to Inference Mode")
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False

    return autoencoder
