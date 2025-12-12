"""Defines a surrogate combining an autoencoder and emulator."""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from src.components.autoencoder import Autoencoder
from src.components.emulator import Emulator
from src.data_processing import Processing


class AutoencoderEmulatorSurrogate:
    """Surrogate model using Autoencoder + Emulator."""

    def benchmark(self) -> dict[str, object]:
        """Benchmark the surrogate model."""
        return {"type": "ae_emulator", "results": "placeholder"}


class Inference:
    """Production inference interface for encoding, decoding, and emulating chemical abundances."""

    def __init__(
        self,
        general_config: DictConfig,
        processing_functions: Processing,
        autoencoder: Autoencoder | None = None,
        emulator: Emulator | None = None,
    ) -> None:
        """Initialize Inference."""
        self.device = general_config.device
        self.autoencoder = autoencoder
        self.emulator = emulator

        self.inverse_abundances_scaling = (
            processing_functions.inverse_abundances_scaling
        )
        self.physical_parameter_scaling = (
            processing_functions.physical_parameter_scaling
        )
        self.latents_scaling = processing_functions.latents_scaling
        self.inverse_latents_scaling = processing_functions.inverse_latents_scaling

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

    def _flatten_latents(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int] | None]:
        """Flatten 3D latents to 2D for decoding."""
        if latents.ndim != 3:
            return latents, None
        batch_size, time_steps, latent_dim = latents.shape
        original_shape = (batch_size, time_steps, -1)
        flat = latents.view(batch_size * time_steps, latent_dim)
        return flat, original_shape

    def _decode_flat_latents(self, flat_latents: torch.Tensor) -> torch.Tensor:
        """Decode 2D latents into abundance space."""
        with torch.no_grad():
            latents_t = self.convert_to_tensor(flat_latents)
            scaled = self.autoencoder.decode(latents_t)
            abundances = self.inverse_abundances_scaling(scaled)
        if isinstance(abundances, torch.Tensor):
            return abundances
        return torch.from_numpy(abundances)

    def _restore_abundance_shape(
        self, abundances: torch.Tensor, original_shape: tuple[int, int, int] | None
    ) -> torch.Tensor:
        """Restore abundances to their original shape if needed."""
        if original_shape is None:
            return abundances
        return abundances.view(*original_shape)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to chemical abundances."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder model not initialized")
        flat_latents, original_shape = self._flatten_latents(latents)
        abundances = self._decode_flat_latents(flat_latents)
        return self._restore_abundance_shape(abundances, original_shape)

    def latent_emulate(
        self, phys: np.ndarray | torch.Tensor, latents: np.ndarray | torch.Tensor
    ) -> torch.Tensor:
        """Predict latent space evolution given physical parameters."""
        if self.emulator is None:
            raise ValueError("Emulator model not initialized")
        with torch.no_grad():
            phys_t = self.convert_to_tensor(phys)
            latents_t = self.convert_to_tensor(latents)
            scaled_latents = self.latents_scaling(latents_t)
            scaled_evolved_latents = self.emulator(phys_t, scaled_latents)
            return self.inverse_latents_scaling(scaled_evolved_latents)

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
