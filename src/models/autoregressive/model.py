"""Autoregressive model for direct chemical abundance evolution prediction."""

import os

import torch
import torch.nn as nn

from src.datasets import DatasetConfig
from src.models.autoregressive.config import AutoregressiveConfig


class Autoregressive(nn.Module):
    """Neural network for abundance-space autoregressive prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """Initialize abundance autoregressive model with a residual MLP cell."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, phys: torch.Tensor, abundances: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequential abundance-space evolution prediction."""
        batch_size, timesteps, _ = phys.shape
        num_species = abundances.shape[1]
        outputs = torch.empty(
            batch_size,
            timesteps,
            num_species,
            device=abundances.device,
            dtype=abundances.dtype,
        )

        for timestep in range(timesteps):
            current_phys = phys[:, timestep, :]
            inputs = torch.cat([current_phys, abundances], dim=1)
            abundances = torch.clamp(abundances + self.net(inputs), 0.0, 1.0)
            outputs[:, timestep, :] = abundances

        return outputs


def load_autoregressive(
    model_class: type[Autoregressive],
    general_config: DatasetConfig,
    ar_config: AutoregressiveConfig,
    inference: bool = False,
) -> Autoregressive:
    """Load abundance autoregressive model with optional pretrained weights."""
    model = model_class(
        input_dim=ar_config.input_dim,
        output_dim=ar_config.output_dim,
        hidden_dim=ar_config.hidden_dim,
        dropout=ar_config.dropout,
    ).to(general_config.device)

    if os.path.exists(ar_config.pretrained_model_path):
        print("Loading Pretrained Model")
        model.load_state_dict(
            torch.load(
                ar_config.pretrained_model_path,
                map_location=torch.device("cpu"),
            )
        )

    if inference:
        print("Setting Autoregressive to Inference Mode")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model
