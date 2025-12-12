"""Provides preprocessing and inverse-scaling utilities for model I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from src.surrogates.autoencoder_emulator import Inference


def load_3d_tensors(cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Load training and validation 3D tensors from disk."""
    from pathlib import Path

    input_dir = Path(cfg.input_dir)
    train_path = input_dir / "initial_train_preprocessed.pt"
    val_path = input_dir / "initial_val_preprocessed.pt"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found. Expected files:\n"
            f"  - {train_path}\n"
            f"  - {val_path}\n"
            f"Run initial preprocessing first: python preprocess.py grav initial"
        )

    training = torch.load(train_path)
    validation = torch.load(val_path)

    print(f"Loaded training tensor: {training.shape}")
    print(f"Loaded validation tensor: {validation.shape}")

    return training, validation


class Processing:
    """Handles scaling and inverse scaling for physical parameters, abundances, and latents."""

    def __init__(
        self,
        dataset_cfg: DictConfig,
        device: str,
        autoencoder_cfg: DictConfig | None = None,
    ) -> None:
        """Initialize Processing."""
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

        if autoencoder_cfg is not None:
            latents_minmax = np.load(autoencoder_cfg.latents_minmax_path)
            print(f"Latents MinMax: {latents_minmax[0]}, {latents_minmax[1]}")
            self.latents_min = torch.tensor(
                latents_minmax[0], dtype=torch.float32, device=self.device
            )
            self.latents_max = torch.tensor(
                latents_minmax[1], dtype=torch.float32, device=self.device
            )

        self.physical_parameter_ranges = dataset_cfg.physical_parameters.ranges
        self.species = dataset_cfg.species
        self.num_species = dataset_cfg.num_species
        self.stoichiometric_matrix_path = dataset_cfg.stoichiometric_matrix_path

    def physical_parameter_scaling(self, physical_parameters: torch.Tensor) -> None:
        """Scale physical parameters into [0, 1] in-place."""
        physical_parameters.log10_()
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)
            physical_parameters[:, i] = (physical_parameters[:, i] - log_param_min) / (
                log_param_max - log_param_min
            )

    def abundances_scaling(self, abundances: torch.Tensor) -> None:
        """Scale abundances into [0, 1] in-place."""
        abundances.log10_()
        abundances.sub_(self.abundances_min)
        abundances.div_(self.abundances_max - self.abundances_min)

    def latents_scaling(self, latents: torch.Tensor) -> torch.Tensor:
        """Scale latents into [0, 1]."""
        return (latents - self.latents_min) / (
            self.latents_max - self.latents_min
        )

    def inverse_physical_parameter_scaling(
        self, physical_parameters: np.ndarray
    ) -> None:
        """Reverse physical parameter scaling in-place."""
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
        """Invert abundance scaling for tensors."""
        log_abundances = abundances * (max_ - min_) + min_
        return torch.exp(exponential_ * log_abundances)

    def inverse_abundances_scaling(
        self, abundances: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Reverse abundance scaling for tensors or ndarrays."""
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
    def _jit_inverse_latents_scaling(
        scaled_latents: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor
    ) -> torch.Tensor:
        """Invert latent scaling for tensors."""
        return scaled_latents * (max_ - min_) + min_

    def inverse_latents_scaling(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        """Reverse latent scaling."""
        return self._jit_inverse_latents_scaling(
            scaled_latents, self.latents_min, self.latents_max
        )

    def save_latents_minmax(
        self,
        autoencoder_cfg: DictConfig,
        dataset_t: torch.Tensor,
        inference_functions: Inference,
    ) -> None:
        """Compute and save min/max values of encoded latents."""
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
