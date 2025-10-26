"""Defines the Autoencoder and loading it."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """Autoencoder model for dimensionality reduction."""

    def __init__(
        self,
        input_dim=333,
        latent_dim=12,
        hidden_dims=(320, 160),
        noise=0.1,
        dropout=0.0,
    ):
        super(Autoencoder, self).__init__()

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
        self.final_activation = nn.Sigmoid()  # For 0-1 bounded output
        self.dropout = nn.Dropout(dropout)
        self.noise = noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the input into the latent space."""
        x = self.activation(self.encoder_bn1(self.encoder_fc1(x)))
        x = self.activation(self.encoder_bn2(self.encoder_fc2(x)))
        x = self.dropout(x)
        z = self.activation(self.encoder_norm3(self.encoder_fc3(x)))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes the latent representation back to the input space."""
        z = F.linear(z, self.encoder_fc3.weight.t()) + self.decoder_bias1
        z = self.activation(self.decoder_bn1(z))

        z = F.linear(z, self.encoder_fc2.weight.t()) + self.decoder_bias2
        z = self.activation(self.decoder_bn2(z))
        z = self.dropout(z)

        x_reconstructed = F.linear(z, self.encoder_fc1.weight.t()) + self.decoder_bias3
        x_reconstructed = self.final_activation(x_reconstructed)
        return x_reconstructed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the autoencoder."""
        z = self.encode(x)
        if self.training and self.noise > 0:
            noise = torch.randn_like(z) * self.noise
            z += noise
        x_reconstructed = self.decode(z)
        return x_reconstructed


def load_autoencoder(
    Autoencoder: type[Autoencoder], GeneralConfig, model_config, inference=False
):
    """Loads the autoencoder model with the given configuration."""
    autoencoder = Autoencoder(
        input_dim=model_config.input_dim,
        latent_dim=model_config.latent_dim,
        hidden_dims=model_config.hidden_dims,
        noise=model_config.noise,
        dropout=model_config.dropout,
    ).to(GeneralConfig.device)
    if os.path.exists(model_config.pretrained_model_path):
        print("Loading Pretrained Model")
        autoencoder.load_state_dict(
            torch.load(
                model_config.pretrained_model_path, map_location=torch.device("cpu")
            )
        )

    if inference:
        print("Setting Autoencoder to Inference Mode")
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False

    return autoencoder
