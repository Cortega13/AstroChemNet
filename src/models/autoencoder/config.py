"""Configuration dataclass for autoencoder model."""

from dataclasses import dataclass, field

from src.datasets import DatasetConfig, DatasetName


@dataclass
class AEConfig:
    """Base autoencoder configuration."""

    dataset_config: DatasetConfig

    columns: list[str] = field(init=False)
    num_columns: int = field(init=False)
    input_dim: int = field(init=False)
    latents_minmax_path: str = field(init=False)
    pretrained_model_path: str = field(init=False)
    save_model_path: str = field(init=False)

    hidden_dims: tuple[int, ...] = (160, 80)
    latent_dim: int = 14

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
    save_model: bool = True

    def __post_init__(self) -> None:
        """Initialize derived attributes from dataset config."""
        self.columns = self.dataset_config.species
        self.num_columns = len(self.columns)
        self.input_dim = self.dataset_config.num_species
        self.latents_minmax_path = self.dataset_config.model_path(
            "autoencoder", "latents_minmax.npy"
        )
        self.pretrained_model_path = self.dataset_config.model_path(
            "autoencoder", "model.pth"
        )
        self.save_model_path = self.pretrained_model_path


@dataclass
class CarboxAEConfig(AEConfig):
    """Autoencoder configuration for Carbox."""

    artifact_name: str = "autoencoder_carbox"
    hidden_dims: tuple[int, ...] = (80, 40)
    latent_dim: int = 7


AE_CONFIGS: dict[DatasetName, type[AEConfig]] = {
    DatasetName.UCLCHEM_GRAV: AEConfig,
    DatasetName.CARBOX_GRAV: CarboxAEConfig,
}


def build_config(dataset_config: DatasetConfig, **overrides) -> AEConfig:
    """Build autoencoder config for a dataset."""
    dataset_name = DatasetName(dataset_config.dataset_name)
    config_cls = AE_CONFIGS[dataset_name]
    return config_cls(dataset_config=dataset_config, **overrides)
