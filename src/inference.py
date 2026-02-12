"""Inference functions for encoding, decoding, and emulating chemical abundances."""

from typing import Optional

import numpy as np
import torch

from configs.general import GeneralConfig
from models.autoencoder import Autoencoder
from models.emulator import Emulator


class Inference:
    """Inference class for encoding, decoding, and emulating chemical abundances."""

    def __init__(
        self,
        GeneralConfig: GeneralConfig,
        processing_functions,
        autoencoder: Optional[Autoencoder] = None,
        emulator: Optional[Emulator] = None,
    ) -> None:
        """Initialize Inference with config, processing functions, and models."""
        self.device = GeneralConfig.device
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

    def convert_to_tensor(self, inputs) -> torch.Tensor:
        """Convert numpy array or other input to torch tensor on device."""
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).float().to(self.device)
        elif isinstance(inputs, torch.Tensor):
            return inputs.float().to(self.device)
        return torch.tensor(inputs, dtype=torch.float32, device=self.device)

    def encode(self, abundances) -> torch.Tensor:
        """Encode abundances to latent space using autoencoder."""
        assert self.autoencoder is not None, "Autoencoder is required for encoding"
        with torch.no_grad():
            abundances = self.convert_to_tensor(abundances)
            latents = self.autoencoder.encode(abundances)
            return latents

    def decode(self, latents) -> torch.Tensor:
        """Decode latent components to abundances using autoencoder."""
        assert self.autoencoder is not None, "Autoencoder is required for decoding"
        reshaped = latents.ndim == 3
        if reshaped:
            B, T, L = latents.shape
            latents = latents.view(B * T, L)

        with torch.no_grad():
            latents = self.convert_to_tensor(latents)
            scaled = self.autoencoder.decode(latents)
            abundances = self.inverse_abundances_scaling(scaled)

        return abundances.view(B, T, -1) if reshaped else abundances

    def latent_emulate(self, phys, latents) -> torch.Tensor:
        """Emulate evolution of latent components using emulator."""
        assert self.emulator is not None, "Emulator is required for latent emulation"
        with torch.no_grad():
            phys = self.convert_to_tensor(phys)
            latents = self.convert_to_tensor(latents)
            scaled_latents = self.latent_components_scaling(latents)
            scaled_evolved_latents = self.emulator(phys, scaled_latents)
            evolved_latents = self.inverse_latent_components_scaling(
                scaled_evolved_latents
            )
            return evolved_latents

    def emulate(self, phys, abundances, skip_encoder: bool = False) -> torch.Tensor:
        """Emulate evolution of abundances using autoencoder and emulator."""
        assert self.autoencoder is not None, "Autoencoder is required for emulation"
        assert self.emulator is not None, "Emulator is required for emulation"
        latents = abundances if skip_encoder else self.encode(abundances)
        evolved_latents = self.latent_emulate(phys, latents)
        evolved_abundances = self.decode(evolved_latents)
        return evolved_abundances
