"""Defines the Emulator model and loading it."""

import os

import torch
import torch.nn as nn


class Emulator(nn.Module):
    """Autoregressively evolves the latent abundances."""

    def __init__(self, input_dim=18, output_dim=14, hidden_dim=32, dropout=0.0) -> None:
        """Initialize Emulator."""
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
        """Applies a forward-pass using the physical parameters and latent variables."""
        batch_size, time_steps, _ = phys.shape
        latent_dim = latents.shape[1]
        outputs = torch.empty(
            batch_size,
            time_steps,
            latent_dim,
            device=latents.device,
            dtype=latents.dtype,
        )

        for t in range(time_steps):
            current_phys = phys[:, t, :]  # [B, P]
            input_tensor = torch.cat([current_phys, latents], dim=1)  # [B, P+L]

            update = self.net(input_tensor)  # [B, L]
            latents = latents + update  # gated residual

            outputs[:, t, :] = latents

        return outputs


def load_emulator(
    emulator_cls: type[Emulator], general_config, model_config, inference=False
):
    """Loads the emulator model with the given configuration."""
    emulator = emulator_cls(
        input_dim=model_config.input_dim,
        output_dim=model_config.output_dim,
        hidden_dim=model_config.hidden_dim,
    ).to(general_config.device)
    if os.path.exists(model_config.pretrained_model_path):
        print("Loading Pretrained Model")
        emulator.load_state_dict(
            torch.load(
                model_config.pretrained_model_path, map_location=torch.device("cpu")
            )
        )
    if inference:
        print("Setting Emulator to Inference Mode")
        emulator.eval()
        for param in emulator.parameters():
            param.requires_grad = False
    return emulator
