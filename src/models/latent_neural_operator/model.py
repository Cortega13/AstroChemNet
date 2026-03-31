"""Latent neural operator model for chemical abundance evolution prediction."""

import os

import torch
import torch.nn as nn

from src.datasets import DatasetConfig
from src.models.latent_neural_operator.config import LatentNeuralOperatorConfig


class LatentOperatorBlock(nn.Module):
    """Residual operator block with latent-token and channel mixing."""

    def __init__(
        self,
        latent_dim: int,
        width: int,
        token_mixing_expansion: int,
        channel_mixing_expansion: int,
        dropout: float,
    ) -> None:
        """Initialize token and channel mixing block."""
        super().__init__()
        self.token_hidden_dim = max(latent_dim * token_mixing_expansion, latent_dim)
        self.channel_hidden_dim = max(width * channel_mixing_expansion, width)
        self.token_norm = nn.LayerNorm(width)
        self.token_in = nn.Linear(latent_dim, self.token_hidden_dim)
        self.token_out = nn.Linear(self.token_hidden_dim, latent_dim)
        self.channel_norm = nn.LayerNorm(width)
        self.channel_in = nn.Linear(width, self.channel_hidden_dim)
        self.channel_out = nn.Linear(self.channel_hidden_dim, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mix information across latent tokens and token channels."""
        mixed_tokens = self.token_norm(tokens).transpose(1, 2)
        mixed_tokens = self.token_in(mixed_tokens)
        mixed_tokens = self.activation(mixed_tokens)
        mixed_tokens = self.dropout(mixed_tokens)
        mixed_tokens = self.token_out(mixed_tokens)
        tokens = tokens + self.dropout(mixed_tokens).transpose(1, 2)

        mixed_channels = self.channel_norm(tokens)
        mixed_channels = self.channel_in(mixed_channels)
        mixed_channels = self.activation(mixed_channels)
        mixed_channels = self.dropout(mixed_channels)
        mixed_channels = self.channel_out(mixed_channels)
        return tokens + self.dropout(mixed_channels)


class LatentNeuralOperator(nn.Module):
    """Causal latent neural operator using residual latent-token mixing."""

    def __init__(
        self,
        latent_dim: int,
        phys_dim: int,
        operator_width: int = 64,
        num_operator_layers: int = 3,
        token_mixing_expansion: int = 2,
        channel_mixing_expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize latent neural operator model."""
        super().__init__()
        self.latent_dim = latent_dim
        self.token_projection = nn.Linear(1, operator_width)
        self.phys_projection = nn.Sequential(
            nn.Linear(phys_dim, operator_width),
            nn.GELU(),
            nn.Linear(operator_width, operator_width),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, latent_dim, operator_width)
        )
        self.blocks = nn.ModuleList(
            [
                LatentOperatorBlock(
                    latent_dim=latent_dim,
                    width=operator_width,
                    token_mixing_expansion=token_mixing_expansion,
                    channel_mixing_expansion=channel_mixing_expansion,
                    dropout=dropout,
                )
                for _ in range(num_operator_layers)
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(operator_width),
            nn.Linear(operator_width, 1),
        )

        output_layer = self.output_head[-1]
        if isinstance(output_layer, nn.Linear):
            nn.init.zeros_(output_layer.bias)
            output_layer.weight.data.mul_(1e-2)

    def forward(self, phys: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Roll the latent state forward across a sequence of physical controls."""
        batch_size, timesteps, _ = phys.shape
        outputs = torch.empty(
            batch_size,
            timesteps,
            self.latent_dim,
            device=latents.device,
            dtype=latents.dtype,
        )

        current = latents
        for timestep in range(timesteps):
            tokens = self.token_projection(current.unsqueeze(-1))
            tokens = tokens + self.phys_projection(phys[:, timestep, :]).unsqueeze(1)
            tokens = tokens + self.position_embedding

            for block in self.blocks:
                tokens = block(tokens)

            delta = self.output_head(tokens).squeeze(-1)
            current = torch.clamp(current + delta, 0.0, 1.0)
            outputs[:, timestep, :] = current

        return outputs


def load_latent_neural_operator(
    model_class: type[LatentNeuralOperator],
    general_config: DatasetConfig,
    model_config: LatentNeuralOperatorConfig,
    inference: bool = False,
) -> LatentNeuralOperator:
    """Load latent neural operator model with optional pretrained weights."""
    model = model_class(
        latent_dim=model_config.output_dim,
        phys_dim=general_config.num_phys,
        operator_width=model_config.operator_width,
        num_operator_layers=model_config.num_operator_layers,
        token_mixing_expansion=model_config.token_mixing_expansion,
        channel_mixing_expansion=model_config.channel_mixing_expansion,
        dropout=model_config.dropout,
    ).to(general_config.device)

    if os.path.exists(model_config.pretrained_model_path):
        print("Loading Pretrained Model")
        model.load_state_dict(
            torch.load(
                model_config.pretrained_model_path,
                map_location=torch.device("cpu"),
            )
        )

    if inference:
        print("Setting LatentNeuralOperator to Inference Mode")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model
