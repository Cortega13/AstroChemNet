"""Configuration dataclass for autoencoder model."""

import os
from dataclasses import dataclass, field

from src.configs.general import GeneralConfig


@dataclass
class AEConfig:
    """Autoencoder model configuration with hyperparameters and paths.

    Args:
        general_config: GeneralConfig instance for dataset-specific settings
    """

    general_config: GeneralConfig = field(default_factory=lambda: GeneralConfig())

    # Derived from general_config (set in __post_init__)
    columns: list[str] = field(init=False)
    num_columns: int = field(init=False)
    input_dim: int = field(init=False)
    latents_minmax_path: str = field(init=False)
    pretrained_model_path: str = field(init=False)
    save_model_path: str = field(init=False)

    # Model architecture
    hidden_dims: tuple[int, ...] = (160, 80)
    latent_dim: int = 14

    # Hyperparameters
    lr: float = 1e-3
    lr_decay: float = 0.5
    lr_decay_patience: int = 12
    betas: tuple[float, float] = (0.99, 0.999)
    weight_decay: float = 1e-4
    power_weight: int = 20
    conservation_weight: float = 1e2
    batch_size: int = 8 * 8192
    stagnant_epoch_patience: int = 20
    gradient_clipping: float = 2
    dropout_decay_patience: int = 10
    dropout_reduction_factor: float = 0.05
    dropout: float = 0.3
    noise: float = 0.1
    shuffle_chunk_size: int = 1

    # Save settings
    save_model: bool = True

    def __post_init__(self) -> None:
        """Initialize derived attributes from general_config."""
        self.columns = self.general_config.species
        self.num_columns = len(self.columns)
        self.input_dim = self.general_config.num_species  # input_dim = output_dim

        # Set up paths using preprocessing directory
        self.latents_minmax_path = os.path.join(
            self.general_config.preprocessing_dir, "latents_minmax.npy"
        )
        self.pretrained_model_path = os.path.join(
            self.general_config.project_root, "outputs/weights/autoencoder.pth"
        )
        self.save_model_path = os.path.join(
            self.general_config.project_root, "outputs/weights/autoencoder.pth"
        )
