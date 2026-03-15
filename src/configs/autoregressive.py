"""Configuration dataclass for abundance-space autoregressive model."""

import os
from dataclasses import dataclass, field

from src.configs.datasets import DatasetConfig


@dataclass
class AutoregressiveConfig:
    """Autoregressive model configuration with hyperparameters and paths."""

    dataset_config: DatasetConfig

    # Derived from configs (set in __post_init__)
    columns: list[str] = field(init=False)
    num_columns: int = field(init=False)
    input_dim: int = field(init=False)
    output_dim: int = field(init=False)
    pretrained_model_path: str = field(init=False)
    save_model_path: str = field(init=False)

    # Model architecture
    hidden_dim: int = 80
    window_size: int = 240

    # Hyperparameters
    lr: float = 1e-3
    lr_decay: float = 0.5
    lr_decay_patience: int = 5
    betas: tuple[float, float] = (0.9, 0.995)
    weight_decay: float = 1e-3
    power_weight: int = 20
    conservation_weight: float = 5e2
    batch_size: int = 6 * 512
    stagnant_epoch_patience: int = 20
    gradient_clipping: float = 1.0
    dropout_decay_patience: int = 3
    dropout_reduction_factor: float = 0.05
    dropout: float = 0.0
    shuffle: bool = True
    shuffle_chunk_size: int = 1

    # Save settings
    save_model: bool = False

    def __post_init__(self) -> None:
        """Initialize derived attributes from configs."""
        self.columns = (
            self.dataset_config.metadata
            + self.dataset_config.phys
            + self.dataset_config.species
        )
        self.num_columns = len(self.columns)
        self.input_dim = self.dataset_config.num_phys + self.dataset_config.num_species
        self.output_dim = self.dataset_config.num_species

        self.pretrained_model_path = os.path.join(
            self.dataset_config.weights_dir, "autoregressive.pth"
        )
        self.save_model_path = self.pretrained_model_path
