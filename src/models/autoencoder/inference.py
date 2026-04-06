"""Autoencoder inference helpers."""

import numpy as np
import torch

from src.datasets import DatasetConfig
from src.models.autoencoder.model import Autoencoder

TensorLike = np.ndarray | torch.Tensor


class Inference:
    """Inference helpers for trained autoencoders."""

    def __init__(
        self,
        general_config: DatasetConfig,
        processing_functions: object,
        model: Autoencoder,
    ) -> None:
        """Store shared inference dependencies."""
        self.general_config = general_config
        self.processing_functions = processing_functions
        self.model = model

    def _to_tensor(self, values: TensorLike) -> torch.Tensor:
        """Convert values to a tensor on the configured device."""
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        return values.to(self.general_config.device)

    def encode(self, values: TensorLike) -> torch.Tensor:
        """Encode abundance values into latent space."""
        return self.model.encode(self._to_tensor(values))

    def decode(self, values: TensorLike) -> torch.Tensor:
        """Decode latent values back into abundance space."""
        return self.model.decode(self._to_tensor(values))
