"""Data preprocessing and postprocessing utilities for abundance scaling and transformations."""

import gc
import re
from typing import Final

import numpy as np
import torch
from numba import njit
from omegaconf import DictConfig

from .inference import Inference

# Chemical elements tracked for stoichiometric matrix
CHEMICAL_ELEMENTS: Final[list[str]] = ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]

# Regex patterns for element matching in species names
ELEMENT_PATTERNS: Final[dict[str, re.Pattern]] = {
    "H": re.compile(r"H(?!E)(\d*)"),
    "HE": re.compile(r"HE(\d*)"),
    "C": re.compile(r"C(?!L)(\d*)"),
    "N": re.compile(r"N(\d*)"),
    "O": re.compile(r"O(\d*)"),
    "S": re.compile(r"S(?!I)(\d*)"),
    "SI": re.compile(r"SI(\d*)"),
    "MG": re.compile(r"MG(\d*)"),
    "CL": re.compile(r"CL(\d*)"),
}


class Processing:
    """Handles scaling and inverse scaling transformations for physical parameters and abundances."""

    def __init__(
        self,
        dataset_cfg: DictConfig,
        device: str,
        autoencoder_cfg: DictConfig | None = None,
    ) -> None:
        self.device = device
        self.exponential = torch.log(torch.tensor(10, device=self.device).float())

        self.abundances_min = torch.tensor(
            np.log10(dataset_cfg.abundances_clipping.lower),
            dtype=torch.float32,
            device=self.device,
        )
        self.abundances_max = torch.tensor(
            np.log10(dataset_cfg.abundances_clipping.upper),
            dtype=torch.float32,
            device=self.device,
        )
        self.abundances_min_np = self.abundances_min.cpu().numpy()
        self.abundances_max_np = self.abundances_max.cpu().numpy()

        if autoencoder_cfg is not None:
            latents_minmax = np.load(autoencoder_cfg.latents_minmax_path)
            print(f"Latents MinMax: {latents_minmax[0]}, {latents_minmax[1]}")
            self.components_min = torch.tensor(
                latents_minmax[0], dtype=torch.float32, device=self.device
            )
            self.components_max = torch.tensor(
                latents_minmax[1], dtype=torch.float32, device=self.device
            )

        self.physical_parameter_ranges = dataset_cfg.physical_parameter_ranges
        self.species = dataset_cfg.species
        self.num_species = dataset_cfg.num_species
        self.stoichiometric_matrix_path = dataset_cfg.stoichiometric_matrix_path

    def physical_parameter_scaling(self, physical_parameters: np.ndarray) -> None:
        """Apply log10 and minmax scaling to physical parameters [0, 1]."""
        np.log10(physical_parameters, out=physical_parameters)
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)

            physical_parameters[:, i] = (physical_parameters[:, i] - log_param_min) / (
                log_param_max - log_param_min
            )

    def abundances_scaling(self, abundances: np.ndarray) -> None:
        """Abundances are log10'd and then minmax scaled to be within [0, 1]."""
        np.log10(abundances, out=abundances)
        np.subtract(abundances, self.abundances_min_np, out=abundances)
        np.divide(
            abundances,
            (self.abundances_max_np - self.abundances_min_np),
            out=abundances,
        )

    def latent_components_scaling(self, components: torch.Tensor) -> torch.Tensor:
        """Scale latent components to [0, 1]."""
        return (components - self.components_min) / (
            self.components_max - self.components_min
        )

    def inverse_physical_parameter_scaling(
        self, physical_parameters: np.ndarray
    ) -> None:
        """Reverses the minmax scaling of the physical parameters."""
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
    def _jit_inverse_abundances_scaling(
        abundances: torch.Tensor,
        min_: torch.Tensor,
        max_: torch.Tensor,
        exponential_: torch.Tensor,
    ) -> torch.Tensor:
        """JIT-compiled inverse abundance scaling for GPU efficiency."""
        log_abundances = abundances * (max_ - min_) + min_
        return torch.exp(exponential_ * log_abundances)

    def inverse_abundances_scaling(
        self, abundances: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Reverse minmax scaling of abundances (handles both Tensor and ndarray)."""
        if isinstance(abundances, torch.Tensor):
            return self._jit_inverse_abundances_scaling(
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
        return abundances

    @staticmethod
    @torch.jit.script
    def _jit_inverse_latent_component_scaling(
        scaled_components: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor
    ) -> torch.Tensor:
        """JIT-compiled inverse latent component scaling."""
        return scaled_components * (max_ - min_) + min_

    def inverse_latent_components_scaling(
        self, scaled_components: torch.Tensor
    ) -> torch.Tensor:
        """Reverse latent component scaling."""
        return self._jit_inverse_latent_component_scaling(
            scaled_components, self.components_min, self.components_max
        )

    def save_latents_minmax(
        self,
        autoencoder_cfg: DictConfig,
        dataset_t: torch.Tensor,
        inference_functions: Inference,
    ) -> None:
        """Compute and save min/max values of encoded latents for normalization."""
        min_, max_ = float("inf"), float("-inf")

        with torch.no_grad():
            for i in range(0, len(dataset_t), autoencoder_cfg.batch_size):
                batch = dataset_t[i : i + autoencoder_cfg.batch_size].to(self.device)
                encoded = inference_functions.encode(batch).cpu()
                min_ = min(min_, encoded.min().item())
                max_ = max(max_, encoded.max().item())

        minmax_np = np.array([min_, max_], dtype=np.float32)
        print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")
        np.save(autoencoder_cfg.latents_minmax_path, minmax_np)

    def build_stoichiometric_matrix(self) -> np.ndarray:
        """Build stoichiometric matrix S where x @ S yields elemental abundances.

        Returns (num_species, num_elements) matrix.

        Note: Excludes electrons (not conserved by UCLCHEM) and SURFACE/BULK species.
        """
        stoichiometric_matrix = np.zeros((len(CHEMICAL_ELEMENTS), self.num_species))
        modified_species = [s.replace("@", "").replace("#", "") for s in self.species]

        for elem_idx, (element, pattern) in enumerate(ELEMENT_PATTERNS.items()):
            for species_idx, species in enumerate(modified_species):
                if species in ["SURFACE", "BULK"]:
                    continue
                match = pattern.search(species)
                if match:
                    multiplier = int(match.group(1)) if match.group(1) else 1
                    stoichiometric_matrix[elem_idx, species_idx] = multiplier

        np.save(self.stoichiometric_matrix_path, stoichiometric_matrix.T)
        return stoichiometric_matrix.T
