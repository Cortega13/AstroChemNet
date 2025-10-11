"""
For simplicity, here we define an Inference script which"""

import numpy as np
import torch


class Inference:
    def __init__(
        self, GeneralConfig, processing_functions, autoencoder=None, emulator=None
    ):
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

    def convert_to_tensor(self, inputs):
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).float().to(self.device)
        elif isinstance(inputs, torch.Tensor):
            return inputs.float().to(self.device)
        return torch.tensor(inputs, dtype=torch.float32, device=self.device)

    def encode(self, abundances):
        with torch.no_grad():
            abundances = self.convert_to_tensor(abundances)
            latents = self.autoencoder.encode(abundances)
            return latents

    def decode(self, latents):
        reshaped = latents.ndim == 3
        if reshaped:
            B, T, L = latents.shape
            latents = latents.view(B * T, L)

        with torch.no_grad():
            latents = self.convert_to_tensor(latents)
            scaled = self.autoencoder.decode(latents)
            abundances = self.inverse_abundances_scaling(scaled)

        return abundances.view(B, T, -1) if reshaped else abundances

    def latent_emulate(self, phys, latents):
        with torch.no_grad():
            phys = self.convert_to_tensor(phys)
            latents = self.convert_to_tensor(latents)
            scaled_latents = self.latent_components_scaling(latents)
            scaled_evolved_latents = self.emulator(phys, scaled_latents)
            evolved_latents = self.inverse_latent_components_scaling(
                scaled_evolved_latents
            )
            return evolved_latents

    def emulate(self, phys, abundances, skip_encoder=False):
        latents = abundances if skip_encoder else self.encode(abundances)
        evolved_latents = self.latent_emulate(phys, latents)
        evolved_abundances = self.decode(evolved_latents)
        return evolved_abundances
