"""Hydra configuration schemas using structured configs."""

from dataclasses import dataclass, field

import numpy as np
from hydra.core.config_store import ConfigStore


@dataclass
class DatasetConfig:
    """Dataset configuration including paths, physical parameters, and species."""

    dataset_path: str
    physical_parameter_ranges: dict[str, tuple[float, float]]
    abundances_lower_clipping: float
    abundances_upper_clipping: float
    metadata: list[str]
    phys: list[str]
    initial_abundances_path: str
    stoichiometric_matrix_path: str
    species_path: str

    # Computed fields populated in __post_init__
    initial_abundances: np.ndarray = field(init=False, repr=False)
    stoichiometric_matrix: np.ndarray = field(init=False, repr=False)
    species: list[str] = field(init=False, repr=False)
    num_metadata: int = field(init=False, repr=False)
    num_phys: int = field(init=False, repr=False)
    num_species: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load arrays and compute derived fields."""
        self.initial_abundances = np.load(self.initial_abundances_path)
        self.stoichiometric_matrix = np.load(self.stoichiometric_matrix_path)
        self.species = np.loadtxt(
            self.species_path, dtype=str, delimiter=" ", comments=None
        ).tolist()

        self.num_metadata = len(self.metadata)
        self.num_phys = len(self.phys)
        self.num_species = len(self.species)


@dataclass
class ModelsConfig:
    """Reusable model configuration for both autoencoder and emulator."""

    model_name: str
    input_dim: int
    lr: float
    lr_decay: float
    lr_decay_patience: int
    betas: tuple[float, float]
    weight_decay: float
    power_weight: float
    conservation_weight: float
    batch_size: int
    stagnant_epoch_patience: int
    gradient_clipping: float
    dropout_decay_patience: int
    dropout_reduction_factor: float
    dropout: float
    shuffle_chunk_size: float
    save_model: bool
    pretrained_model_path: str
    save_model_path: str

    # Optional architecture parameters
    output_dim: int | None = None
    hidden_dim: int | None = None
    hidden_dims: tuple[int, ...] | None = None
    latent_dim: int | None = None
    window_size: int | None = None

    # Optional hyperparameters
    noise: float | None = None
    shuffle: bool | None = None
    latents_minmax_path: str | None = None

    # Computed fields
    columns: list[str] = field(init=False, repr=False, default_factory=list)
    num_columns: int = field(init=False, repr=False, default=0)

    def setup_columns(
        self, metadata: list[str], phys: list[str], species: list[str]
    ) -> None:
        """Compute columns based on dataset config."""
        if self.model_name == "autoencoder":
            self.columns = species
        elif self.model_name == "emulator":
            self.columns = metadata + phys + species
        else:
            self.columns = []

        self.num_columns = len(self.columns)


@dataclass
class Config:
    """Top-level configuration composing dataset and model configs."""

    dataset: DatasetConfig
    model: ModelsConfig
    autoencoder: ModelsConfig | None = (
        None  # Optional second model (for emulator training)
    )

    def __post_init__(self) -> None:
        """Setup model columns based on dataset."""
        self.model.setup_columns(
            self.dataset.metadata, self.dataset.phys, self.dataset.species
        )
        if self.autoencoder is not None:
            self.autoencoder.setup_columns(
                self.dataset.metadata, self.dataset.phys, self.dataset.species
            )


# Register configs with Hydra ConfigStore
cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="datasets", name="base_dataset", node=DatasetConfig)
cs.store(group="models", name="base_model", node=ModelsConfig)
