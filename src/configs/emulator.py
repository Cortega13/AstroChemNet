"""Configuration dataclass for emulator model."""

import os
from dataclasses import dataclass, field

from src.configs.autoencoder import AEConfig
from src.configs.general import GeneralConfig


@dataclass
class EMConfig:
    """Emulator model configuration with hyperparameters and paths.

    Args:
        general_config: GeneralConfig instance for dataset-specific settings
        ae_config: AEConfig instance for autoencoder settings
    """

    general_config: GeneralConfig = field(default_factory=lambda: GeneralConfig())
    ae_config: AEConfig = field(default_factory=lambda: AEConfig())

    # Derived from configs (set in __post_init__)
    columns: list[str] = field(init=False)
    num_columns: int = field(init=False)
    input_dim: int = field(init=False)
    output_dim: int = field(init=False)
    pretrained_model_path: str = field(init=False)
    save_model_path: str = field(init=False)

    # Model architecture
    hidden_dim: int = 180
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
            self.general_config.metadata
            + self.general_config.phys
            + self.general_config.species
        )
        self.num_columns = len(self.columns)
        self.input_dim = self.general_config.num_phys + self.ae_config.latent_dim
        self.output_dim = self.ae_config.latent_dim

        # Set up paths
        self.pretrained_model_path = os.path.join(
            self.general_config.project_root, "outputs/weights/mlp.pth"
        )
        self.save_model_path = os.path.join(
            self.general_config.project_root, "outputs/weights/mlp.pth"
        )
