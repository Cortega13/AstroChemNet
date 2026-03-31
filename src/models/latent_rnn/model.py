"""Latent RNN model for chemical abundance evolution prediction."""

import os

import torch
import torch.nn as nn

from src.datasets import DatasetConfig
from src.models.latent_rnn.config import LatentRNNConfig


class LatentRNN(nn.Module):
    """GRU-based latent rollout model with residual latent updates."""

    def __init__(
        self,
        latent_dim: int,
        phys_dim: int,
        hidden_dim: int = 180,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize latent RNN with recurrent hidden state and residual head."""
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_projection = nn.Sequential(
            nn.Linear(phys_dim + latent_dim, hidden_dim),
            nn.GELU(),
        )
        self.hidden_projection = nn.Linear(latent_dim, hidden_dim)
        self.cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.hidden_dropout = nn.Dropout(dropout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        final_layer = self.output_head[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.bias)
            final_layer.weight.data.mul_(1e-2)

    def forward(self, phys: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Roll the latent state forward across the physical-control sequence."""
        batch_size, timesteps, _ = phys.shape
        outputs = torch.empty(
            batch_size,
            timesteps,
            self.latent_dim,
            device=latents.device,
            dtype=latents.dtype,
        )

        recurrent_states = [self.hidden_projection(latents)] + [
            torch.zeros(
                batch_size,
                self.hidden_dim,
                device=latents.device,
                dtype=latents.dtype,
            )
            for _ in range(self.num_layers - 1)
        ]
        current = latents

        for timestep in range(timesteps):
            layer_input = self.input_projection(
                torch.cat([phys[:, timestep, :], current], dim=1)
            )
            for layer, cell in enumerate(self.cells):
                recurrent_states[layer] = cell(layer_input, recurrent_states[layer])
                layer_input = self.hidden_dropout(recurrent_states[layer])

            delta = self.output_head(torch.cat([layer_input, current], dim=1))
            current = torch.clamp(current + delta, 0.0, 1.0)
            outputs[:, timestep, :] = current

        return outputs


def load_latent_rnn(
    model_class: type[LatentRNN],
    general_config: DatasetConfig,
    model_config: LatentRNNConfig,
    inference: bool = False,
) -> LatentRNN:
    """Load latent RNN model with optional pretrained weights."""
    model = model_class(
        latent_dim=model_config.output_dim,
        phys_dim=general_config.num_phys,
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
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
        print("Setting LatentRNN to Inference Mode")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model
