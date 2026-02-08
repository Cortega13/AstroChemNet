"""Data loading utilities for training and inference."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.configs import AutoencoderConfig, ComponentConfig, DatasetConfig


class ChunkedShuffleSampler(Sampler):
    """Shuffle data in chunks for memory efficiency."""

    def __init__(self, data_size: int, chunk_size: int, seed: int = 13):
        """Initialize ChunkedShuffleSampler."""
        super().__init__()
        self.data_size = int(data_size)
        self.chunk_size = int(chunk_size)
        self.base_seed = seed
        self.epoch = 0

        self.chunks = []
        start = 0
        while start < self.data_size:
            end = min(start + self.chunk_size, self.data_size)
            self.chunks.append((start, end))
            start = end

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for reproducible shuffling."""
        self.epoch = epoch

    def __iter__(self):
        """Iterate over shuffled indices."""
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)

        chunk_indices = torch.randperm(len(self.chunks), generator=g)

        for i, chunk_idx in enumerate(chunk_indices):
            chunk_seed = self.base_seed + self.epoch * 10000 + i
            g.manual_seed(chunk_seed)

            start, end = self.chunks[chunk_idx]
            length = end - start

            chunk_perm = torch.randperm(length, generator=g)
            chunk_perm += start

            yield from chunk_perm.tolist()

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.data_size


class AutoencoderDataset(Dataset):
    """Tensor Dataset for autoencoder training.

    Uses __getitems__ for efficient batch loading (~10^3x speedup).
    """

    def __init__(self, data_matrix: torch.Tensor):
        """Initialize AutoencoderDataset."""
        self.data_matrix = data_matrix
        self.size = len(data_matrix)
        print(f"Data_matrix Memory usage: {data_matrix.nbytes / (1024**2):.3f} MB")

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.size

    def __getitems__(self, indices: list[int]) -> torch.Tensor:
        """Load multiple samples at once for efficient batching."""
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        return self.data_matrix[tensor_indices]


class AutoregressiveDataset(Dataset):
    """Tensor Dataset for emulator training with sliding window approach.

    Computes window indices on-the-fly for memory efficiency, like language models.
    """

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        autoencoder_cfg: AutoencoderConfig,
        data_3d: torch.Tensor,
        horizon: int,
    ):
        """Initialize AutoregressiveDataset with sliding window logic."""
        self.data_3d = data_3d.contiguous()
        self.num_phys = dataset_cfg.num_phys
        self.num_latents = autoencoder_cfg.latent_dim
        self.horizon = horizon

        self.n_tracers, self.n_timesteps, _ = data_3d.shape
        self.windows_per_tracer = self.n_timesteps - horizon
        self.total_windows = self.n_tracers * self.windows_per_tracer

        print(f"Data_matrix Memory usage: {data_3d.nbytes / (1024**2):.3f} MB")
        print(f"Dataset Size: {self.total_windows} windows")
        print(
            f"  - {self.n_tracers} tracers × {self.windows_per_tracer} windows/tracer"
        )
        print(f"  - Horizon: {horizon} timesteps\n")

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.total_windows

    def __getitems__(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load sliding window data for multiple samples efficiently."""
        # Convert to tensor for vectorized operations
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.data_3d.device)

        # Vectorized index calculations
        tracer_indices = idx_tensor // self.windows_per_tracer
        start_times = idx_tensor % self.windows_per_tracer

        batch_size = len(indices)

        # Preallocate output tensors
        batch_phys = torch.empty(
            (batch_size, self.horizon, self.num_phys),
            dtype=self.data_3d.dtype,
            device=self.data_3d.device,
        )
        batch_initial_latents = torch.empty(
            (batch_size, self.num_latents),
            dtype=self.data_3d.dtype,
            device=self.data_3d.device,
        )
        batch_target_latents = torch.empty(
            (batch_size, self.horizon, self.num_latents),
            dtype=self.data_3d.dtype,
            device=self.data_3d.device,
        )

        # Vectorized data extraction
        for i, (tracer_idx, start_t) in enumerate(zip(tracer_indices, start_times)):
            window = self.data_3d[tracer_idx, start_t : start_t + self.horizon + 1]
            batch_phys[i] = window[: self.horizon, : self.num_phys]
            batch_initial_latents[i] = window[0, self.num_phys :]
            batch_target_latents[i] = window[1 : self.horizon + 1, self.num_phys :]

        return batch_phys, batch_initial_latents, batch_target_latents


def tensor_to_dataloader(
    model_cfg: ComponentConfig,
    torch_dataset: AutoencoderDataset | AutoregressiveDataset,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader with chunked shuffling for memory efficiency."""
    sampler = None
    if shuffle:
        data_size = len(torch_dataset)
        multiplier = model_cfg.shuffle_chunk_size
        sampler = ChunkedShuffleSampler(
            data_size, chunk_size=int(multiplier * data_size)
        )

    return DataLoader(
        torch_dataset,
        batch_size=model_cfg.batch_size,
        pin_memory=True,
        num_workers=model_cfg.num_workers,
        shuffle=False if sampler else shuffle,
        sampler=sampler,
    )
