"""Emulator model for chemical abundance evolution prediction."""

import os

import torch
import torch.nn as nn

from src.configs.emulator import EMConfig
from src.configs.general import GeneralConfig


class Emulator(nn.Module):
    """Neural network emulator for predicting chemical evolution in latent space."""

    def __init__(
        self,
        input_dim: int = 18,
        output_dim: int = 14,
        hidden_dim: int = 32,
        dropout: float = 0.0,
    ) -> None:
        """Initialize emulator with MLP network for sequential prediction."""
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


def load_emulator(
    emulator_class: type[Emulator],
    GeneralConfig: GeneralConfig,
    EMConfig: EMConfig,
    inference: bool = False,
) -> Emulator:
    """Load emulator model with optional pretrained weights."""
    emulator = emulator_class(
        input_dim=EMConfig.input_dim,
        output_dim=EMConfig.output_dim,
        hidden_dim=EMConfig.hidden_dim,
    ).to(GeneralConfig.device)
    if os.path.exists(EMConfig.pretrained_model_path):
        print("Loading Pretrained Model")
        emulator.load_state_dict(
            torch.load(EMConfig.pretrained_model_path, map_location=torch.device("cpu"))
        )
    if inference:
        print("Setting Emulator to Inference Mode")
        emulator.eval()
        for param in emulator.parameters():
            param.requires_grad = False
    return emulator
