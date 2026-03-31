"""LatentAR model for chemical abundance evolution prediction."""

import os

import torch
import torch.nn as nn

from src.datasets import DatasetConfig
from src.models.latent_autoregressive.config import ARConfig


class LatentAR(nn.Module):
    """Neural network for latent autoregressive chemical evolution prediction."""

    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 14,
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """Initialize latent autoregressive with MLP network for sequential prediction."""
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

    def forward(self, phys: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequential latent space evolution prediction."""
        batch_size, timesteps, _ = phys.shape
        latent_dim = latents.shape[1]
        outputs = torch.empty(
            batch_size,
            timesteps,
            latent_dim,
            device=latents.device,
            dtype=latents.dtype,
        )

        for timestep in range(timesteps):
            current_phys = phys[:, timestep, :]
            inputs = torch.cat([current_phys, latents], dim=1)
            latents = torch.clamp(latents + self.net(inputs), 0.0, 1.0)
            outputs[:, timestep, :] = latents

        return outputs


def load_latent_autoregressive(
    model_class: type[LatentAR],
    general_config: DatasetConfig,
    ar_config: ARConfig,
    inference: bool = False,
) -> LatentAR:
    """Load latent autoregressive model with optional pretrained weights."""
    model = model_class(
        input_dim=ar_config.input_dim,
        output_dim=ar_config.output_dim,
        hidden_dim=ar_config.hidden_dim,
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
        print("Setting LatentAR to Inference Mode")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model
