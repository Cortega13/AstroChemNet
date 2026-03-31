"""Shared scaling and processing helpers."""

from typing import overload

import numpy as np
import torch

from src.datasets import DatasetConfig
from src.models.autoencoder.config import AEConfig


class Processing:
    """Group of preprocessing and postprocessing functions for data scaling."""

    def __init__(
        self, general_config: DatasetConfig, ae_config: AEConfig | None = None
    ) -> None:
        self.device = general_config.device
        self.exponential = torch.log(torch.tensor(10, device=self.device).float())

        self.abundances_min = torch.tensor(
            np.log10(general_config.abundances_lower_clipping),
            dtype=torch.float32,
            device=self.device,
        )
        self.abundances_max = torch.tensor(
            np.log10(general_config.abundances_upper_clipping),
            dtype=torch.float32,
            device=self.device,
        )
        self.abundances_min_np = self.abundances_min.cpu().numpy()
        self.abundances_max_np = self.abundances_max.cpu().numpy()

        if ae_config is not None:
            latents_minmax = np.load(ae_config.latents_minmax_path)
            print(f"Latents MinMax: {latents_minmax[0]}, {latents_minmax[1]}")
            self.components_min = torch.tensor(
                latents_minmax[0], dtype=torch.float32, device=self.device
            )
            self.components_max = torch.tensor(
                latents_minmax[1], dtype=torch.float32, device=self.device
            )

        self.physical_parameter_ranges = general_config.physical_parameter_ranges
        self.species = general_config.species
        self.num_species = general_config.num_species

    ### PreProcessing Functions

    def physical_parameter_scaling(self, physical_parameters: np.ndarray) -> None:
        """Minmax scale physical parameters to [0, 1] using log10 transformation."""
        np.log10(physical_parameters, out=physical_parameters)
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)

            physical_parameters[:, i] = (physical_parameters[:, i] - log_param_min) / (
                log_param_max - log_param_min
            )

    def jit_abundances_scaling(self, abundances: torch.Tensor) -> torch.Tensor:
        """Log10 transform and minmax scale abundances to [0, 1]."""
        abundances = torch.log10(abundances)
        abundances = (abundances - self.abundances_min) / (
            self.abundances_max - self.abundances_min
        )
        return abundances

    def abundances_scaling(self, abundances: np.ndarray) -> None:
        """Log10 transform and minmax scale abundances to [0, 1] in-place."""
        np.log10(abundances, out=abundances)
        np.subtract(abundances, self.abundances_min_np, out=abundances)
        np.divide(
            abundances,
            (self.abundances_max_np - self.abundances_min_np),
            out=abundances,
        )

    def latent_components_scaling(self, components: torch.Tensor) -> torch.Tensor:
        """Minmax scale latent components to [0, 1]."""
        return (components - self.components_min) / (
            self.components_max - self.components_min
        )

    ### PostProcessing Functions

    def inverse_physical_parameter_scaling(
        self, physical_parameters: np.ndarray
    ) -> None:
        """Reverse minmax scaling of physical parameters in-place."""
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)

            physical_parameters[:, i] = (
                physical_parameters[:, i] * (log_param_max - log_param_min)
                + log_param_min
            )

        np.power(10, physical_parameters, out=physical_parameters)

    @staticmethod
    @torch.jit.script
    def jit_inverse_abundances_scaling(
        abundances: torch.Tensor,
        min_: torch.Tensor,
        max_: torch.Tensor,
        exponential_: torch.Tensor,
    ) -> torch.Tensor:
        """JIT-compiled function to reverse minmax scaling of abundances."""
        log_abundances = abundances * (max_ - min_) + min_
        abundances = torch.exp(exponential_ * log_abundances)
        return abundances

    @overload
    def inverse_abundances_scaling(self, abundances: torch.Tensor) -> torch.Tensor: ...

    @overload
    def inverse_abundances_scaling(self, abundances: np.ndarray) -> None: ...

    def inverse_abundances_scaling(
        self, abundances: torch.Tensor | np.ndarray
    ) -> torch.Tensor | None:
        """Reverse minmax scaling of abundances for torch or numpy arrays."""
        if isinstance(abundances, torch.Tensor):
            return self.jit_inverse_abundances_scaling(
                abundances,
                self.abundances_min,
                self.abundances_max,
                self.exponential,
            )

        ab_min_np = self.abundances_min.cpu().numpy()
        ab_max_np = self.abundances_max.cpu().numpy()
        exponential_np = self.exponential.cpu().numpy()

        np.multiply(abundances, (ab_max_np - ab_min_np), out=abundances)
        np.add(abundances, ab_min_np, out=abundances)
        np.exp(exponential_np * abundances, out=abundances)

        return None

    @staticmethod
    @torch.jit.script
    def jit_inverse_latent_component_scaling(
        scaled_components: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor
    ) -> torch.Tensor:
        """JIT-compiled function to reverse latent component scaling."""
        return scaled_components * (max_ - min_) + min_

    def inverse_latent_components_scaling(
        self, scaled_components: torch.Tensor
    ) -> torch.Tensor:
        """Reverse the minmax scaling of latent components."""
        return self.jit_inverse_latent_component_scaling(
            scaled_components, self.components_min, self.components_max
        )
