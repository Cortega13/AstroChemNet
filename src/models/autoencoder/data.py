"""Autoencoder data helpers."""

import torch
from torch.utils.data import Dataset


class AutoencoderDataset(Dataset):
    """Tensor dataset for autoencoder training."""

    def __init__(self, data_matrix: torch.Tensor) -> None:
        """Initialize the dataset with a data matrix tensor."""
        self.data_matrix = data_matrix
        print(f"Data_matrix Memory usage: {self.data_matrix.nbytes / (1024**2):.3f} MB")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_matrix)

    def __getitems__(self, indices):
        """Load multiple samples at once for efficient batch retrieval."""
        features = self.data_matrix[torch.tensor(indices, dtype=torch.long)]
        return features, 1
