"""LatentAR model for chemical abundance evolution prediction."""

import os

import torch
import torch.nn as nn

from src.configs.datasets import DatasetConfig
from src.configs.latent_autoregressive import ARConfig


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
        # self.gate_layer = nn.Linear(input_dim, output_dim)  # new

    def forward(self, phys: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequential latent space evolution prediction."""
        B, T, P = phys.shape
        L = latents.shape[1]
        outputs = torch.empty(B, T, L, device=latents.device, dtype=latents.dtype)

        for t in range(T):
            current_phys = phys[:, t, :]  # [B, P]
            input = torch.cat([current_phys, latents], dim=1)  # [B, P+L]

            update = self.net(input)  # [B, L]
            # gate = torch.sigmoid(self.gate_layer(input))  # [B, L]
            latents = latents + update  # gated residual

            outputs[:, t, :] = latents

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
                ar_config.pretrained_model_path, map_location=torch.device("cpu")
            )
        )
    if inference:
        print("Setting LatentAR to Inference Mode")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return model
