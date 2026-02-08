"""Defines minimal dataclasses for dataset and model component configs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DatasetConfig:
    """Defines dataset configuration values."""

    name: str
    raw_path: str
    input_key: str
    n_species: int
    species_file: str
    initial_abundances: str
    phys: list[str]
    physical_ranges: dict[str, tuple[float, float]]
    n_params: int
    metadata_columns: list[str]
    columns_to_drop: list[str]
    num_metadata: int
    num_phys: int
    abundances_lower: float
    abundances_upper: float
    stoichiometric_matrix_path: str
    train_split: float
    seed: int
    num_species: int
    species: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutoencoderConfig:
    """Defines autoencoder component configuration values."""

    name: str
    type: str
    dataset: str
    preprocessing_method: str
    hidden_dims: tuple[int, int]
    latent_dim: int
    lr: float
    lr_decay: float
    lr_decay_patience: int
    betas: tuple[float, float]
    weight_decay: float
    batch_size: int
    epochs: int
    stagnant_epoch_patience: int
    gradient_clipping: float
    dropout: float
    dropout_decay_patience: int
    dropout_reduction_factor: float
    noise: float
    shuffle_chunk_size: float
    latents_minmax_path: str | None
    power_weight: float
    conservation_weight: float
    pretrained_model_path: str | None = None
    input_dim: int | None = None
    num_workers: int = 10


@dataclass(slots=True)
class EmulatorConfig:
    """Defines emulator component configuration values."""

    name: str
    type: str
    dataset: str
    preprocessing_method: str
    autoencoder_component: str
    hidden_dim: int
    horizon: int
    lr: float
    lr_decay: float
    lr_decay_patience: int
    betas: tuple[float, float]
    weight_decay: float
    batch_size: int
    epochs: int
    stagnant_epoch_patience: int
    gradient_clipping: float
    dropout: float
    shuffle_chunk_size: float
    power_weight: float
    conservation_weight: float
    num_workers: int = 10


ComponentConfig = AutoencoderConfig | EmulatorConfig


@dataclass(slots=True)
class SurrogateConfig:
    """Defines surrogate benchmark configuration values."""

    name: str
    description: str
    components: dict[str, str]
    rollout_steps: int
    device: str
