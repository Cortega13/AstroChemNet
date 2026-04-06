"""Configuration dataclass for abundance-space autoregressive model."""

from dataclasses import dataclass, field

from src.datasets import DatasetConfig, DatasetName


@dataclass
class AutoregressiveConfig:
    """Base autoregressive configuration."""

    dataset_config: DatasetConfig

    columns: list[str] = field(init=False)
    num_columns: int = field(init=False)
    input_dim: int = field(init=False)
    output_dim: int = field(init=False)
    pretrained_model_path: str = field(init=False)
    save_model_path: str = field(init=False)

    hidden_dim: int = 80
    window_size: int = 240

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
    save_model: bool = False

    def __post_init__(self) -> None:
        """Initialize derived attributes from dataset config."""
        self.columns = (
            self.dataset_config.metadata
            + self.dataset_config.phys
            + self.dataset_config.species
        )
        self.num_columns = len(self.columns)
        self.input_dim = self.dataset_config.num_phys + self.dataset_config.num_species
        self.output_dim = self.dataset_config.num_species
        self.pretrained_model_path = self.dataset_config.model_path("autoregressive", "model.pth")
        self.save_model_path = self.pretrained_model_path


@dataclass
class CarboxAutoregressiveConfig(AutoregressiveConfig):
    """Autoregressive configuration for Carbox."""

    artifact_name: str = "autoregressive_carbox"
    hidden_dim: int = 40
    window_size: int = 240


AR_CONFIGS: dict[DatasetName, type[AutoregressiveConfig]] = {
    DatasetName.UCLCHEM_GRAV: AutoregressiveConfig,
    DatasetName.CARBOX_GRAV: CarboxAutoregressiveConfig,
}


def build_config(dataset_config: DatasetConfig, **overrides) -> AutoregressiveConfig:
    """Build autoregressive config for a dataset."""
    dataset_name = DatasetName(dataset_config.dataset_name)
    config_cls = AR_CONFIGS[dataset_name]
    return config_cls(dataset_config=dataset_config, **overrides)
