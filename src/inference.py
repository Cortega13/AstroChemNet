"""Inference interface for autoencoder and emulator model predictions."""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from .data_processing import Processing
from .models.autoencoder import Autoencoder
from .models.emulator import Emulator


class Inference:
    """Production inference interface for encoding, decoding, and emulating chemical abundances."""

    def __init__(
        self,
        general_config: DictConfig,
        processing_functions: Processing,
        autoencoder: Autoencoder | None = None,
        emulator: Emulator | None = None,
    ) -> None:
        self.device = general_config.device
        self.autoencoder = autoencoder
        self.emulator = emulator

        self.inverse_abundances_scaling = (
            processing_functions.inverse_abundances_scaling
        )
        self.physical_parameter_scaling = (
            processing_functions.physical_parameter_scaling
        )
        self.latent_components_scaling = processing_functions.latent_components_scaling
        self.inverse_latent_components_scaling = (
            processing_functions.inverse_latent_components_scaling
        )

    def convert_to_tensor(
        self, inputs: np.ndarray | torch.Tensor | Any
    ) -> torch.Tensor:
        """Convert various input types to torch.Tensor on the correct device."""
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).float().to(self.device)
        if isinstance(inputs, torch.Tensor):
            return inputs.float().to(self.device)
        return torch.tensor(inputs, dtype=torch.float32, device=self.device)

    def encode(self, abundances: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode chemical abundances to latent space representation."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not initialized")
        with torch.no_grad():
            abundances_t = self.convert_to_tensor(abundances)
            return self.autoencoder.encode(abundances_t)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to chemical abundances."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not initialized")

        reshaped = latents.ndim == 3
        original_shape: tuple[int, int, int] | None = None
        if reshaped:
            B, T, L = latents.shape
            original_shape = (B, T, -1)
            latents = latents.view(B * T, L)

        with torch.no_grad():
            latents_t = self.convert_to_tensor(latents)
            scaled = self.autoencoder.decode(latents_t)
            abundances = self.inverse_abundances_scaling(scaled)

        if original_shape is not None and isinstance(abundances, torch.Tensor):
            return abundances.view(*original_shape)
        return (
            abundances
            if isinstance(abundances, torch.Tensor)
            else torch.from_numpy(abundances)
        )

    def latent_emulate(
        self, phys: np.ndarray | torch.Tensor, latents: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Predict latent space evolution given physical parameters."""
        if self.emulator is None:
            raise ValueError("Emulator model not initialized")
        with torch.no_grad():
            phys_t = self.convert_to_tensor(phys)
            latents_t = self.convert_to_tensor(latents)
            scaled_latents = self.latent_components_scaling(latents_t)
            scaled_evolved_latents = self.emulator(phys_t, scaled_latents)
            return self.inverse_latent_components_scaling(scaled_evolved_latents)

    def emulate(
        self,
        phys: np.ndarray | torch.Tensor,
        abundances: np.ndarray | torch.Tensor,
        skip_encoder: bool = False,
    ) -> torch.Tensor:
        """End-to-end emulation: encode abundances, predict evolution, decode results."""
        latents: torch.Tensor | np.ndarray = (
            abundances if skip_encoder else self.encode(abundances)
        )
        evolved_latents = self.latent_emulate(phys, latents)
        return self.decode(evolved_latents)
