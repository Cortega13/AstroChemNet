"""Data loading utilities for training and inference."""

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler


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


class EmulatorSequenceDataset(Dataset):
    """Tensor Dataset for emulator training with sequence indexing.

    Uses __getitems__ to reuse rows across training elements, reducing memory overhead.
    """

    def __init__(
        self,
        dataset_cfg: DictConfig,
        autoencoder_cfg: DictConfig,
        data_matrix: torch.Tensor,
        data_indices: torch.Tensor,
    ):
        """Initialize EmulatorSequenceDataset."""
        self.data_matrix = data_matrix.contiguous()
        self.data_indices = data_indices.contiguous()
        self.num_datapoints = len(data_indices)
        self.num_metadata = dataset_cfg.num_metadata
        self.num_phys = dataset_cfg.num_phys
        self.num_species = dataset_cfg.num_species
        self.num_latents = autoencoder_cfg.latent_dim

        print(f"Data_matrix Memory usage: {data_matrix.nbytes / (1024**2):.3f} MB")
        print(f"Indices_matrix Memory usage: {data_indices.nbytes / (1024**2):.3f} MB")
        print(f"Dataset Size: {len(data_indices)}\n")

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_datapoints

    def __getitems__(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load sequence data for multiple samples efficiently."""
        data_indices = self.data_indices[indices]
        rows = self.data_matrix[data_indices]

        physical_parameters = rows[
            :, :-1, self.num_metadata : self.num_metadata + self.num_phys
        ]
        features = rows[:, 0, -self.num_latents :]
        targets = rows[:, 1:, self.num_metadata + self.num_phys : -self.num_latents]

        return physical_parameters, features, targets


def tensor_to_dataloader(
    model_cfg: DictConfig,
    torch_dataset: AutoencoderDataset | EmulatorSequenceDataset,
) -> DataLoader:
    """Create a DataLoader with chunked shuffling for memory efficiency."""
    data_size = len(torch_dataset)
    multiplier = model_cfg.shuffle_chunk_size
    sampler = ChunkedShuffleSampler(data_size, chunk_size=int(multiplier * data_size))

    return DataLoader(
        torch_dataset,
        batch_size=model_cfg.batch_size,
        pin_memory=True,
        num_workers=getattr(model_cfg, "num_workers", 10),
        in_order=False,
        sampler=sampler,
    )
