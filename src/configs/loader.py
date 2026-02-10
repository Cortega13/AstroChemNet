"""Builds runtime config objects from in-code dataclass registries."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import torch

from src.configs.types import (
    AutoencoderConfig,
    ComponentConfig,
    DatasetConfig,
    EmulatorConfig,
    PreprocessingConfig,
    PreprocessRunConfig,
    RuntimeConfig,
    RuntimePathsConfig,
    SurrogateConfig,
    TrainingRunConfig,
)

RUNTIME = RuntimeConfig(
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=13,
    paths=RuntimePathsConfig(
        weights_dir="outputs/weights",
        preprocessed_dir="outputs/preprocessed",
    ),
)

uclchem_grav = DatasetConfig(
    name="uclchem_grav",
    raw_path="data/uclchem_grav.h5",
    input_key="df",
    species_file="data/uclchem_species.txt",
    initial_abundances="data/initial_abundances.npy",
    phys_ranges={
        "Density": (68481.0, 1284211415.0),
        "Radfield": (1.0e-4, 26.0),
        "Av": (0.1, 6914.0),
        "gasTemp": (13.0, 133.0),
    },
    metadata_columns=["Index", "Model", "Time"],
    columns_to_drop=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"],
    num_metadata=3,
    num_phys=4,
    num_species=333,
    abundances_lower=1.0e-20,
    abundances_upper=1.0,
    stoichiometric_matrix_path="outputs/preprocessed/uclchem_grav/stoichiometric_matrix.pt",
    train_split=0.75,
)

carbox_grav = DatasetConfig(
    name="carbox_grav",
    raw_path="data/carbox_grav.h5",
    input_key="large",
    species_file="data/carbox_species.txt",
    initial_abundances="data/initial_abundances.npy",
    phys_ranges={
        "Density": (0.5715546011924744, 195699.671875),
        "Radfield": (3.8874722463333455e-07, 3.748945713043213),
        "Av": (0.0075920443050563335, 18.559925079345703),
        "gasTemp": (2.591064214706421, 7209.87451171875),
    },
    metadata_columns=["Index", "Model", "Time"],
    columns_to_drop=[],
    num_metadata=3,
    num_phys=4,
    num_species=163,
    abundances_lower=1.0e-20,
    abundances_upper=1.0,
    stoichiometric_matrix_path=(
        "outputs/preprocessed/carbox_grav/uclchem_grav/stoichiometric_matrix.pt"
    ),
    train_split=0.75,
)

autoencoder_uclchem_grav = AutoencoderConfig(
    trainingtype="autoencoder",
    name="autoencoder_uclchem_grav",
    dataset="uclchem_grav",
    preprocessing_method="autoencoder",
    hidden_dims=(160, 80),
    latent_dim=14,
    lr=1.0e-3,
    lr_decay=0.5,
    lr_decay_patience=12,
    betas=(0.99, 0.999),
    weight_decay=1.0e-4,
    batch_size=65536,
    epochs=1000,
    stagnant_epoch_patience=20,
    gradient_clipping=2.0,
    dropout=0.3,
    dropout_decay_patience=10,
    dropout_reduction_factor=0.1,
    noise=0.1,
    shuffle_chunk_size=0.1,
    latents_minmax_path="outputs/weights/autoencoder_uclchem_grav/latents_minmax.npy",
    power_weight=20.0,
    conservation_weight=100.0,
    pretrained_model_path="outputs/weights/autoencoder_uclchem_grav/weights.pth",
)

emulator_uclchem_grav = EmulatorConfig(
    trainingtype="emulator",
    name="emulator_uclchem_grav",
    dataset="uclchem_grav",
    preprocessing_method="autoregressive",
    autoencoder_component="autoencoder_uclchem_grav",
    hidden_dim=256,
    horizon=5,
    lr=1.0e-3,
    lr_decay=0.5,
    lr_decay_patience=12,
    betas=(0.99, 0.999),
    weight_decay=1.0e-4,
    batch_size=16384,
    epochs=500,
    stagnant_epoch_patience=20,
    gradient_clipping=1.0,
    dropout=0.3,
    shuffle_chunk_size=0.1,
    power_weight=20.0,
    conservation_weight=100.0,
)

uclchem_grav_preprocessing = PreprocessingConfig(
    name="uclchem_grav",
    input_source="uclchem_grav",
    train_tensor="uclchem_grav_train_preprocessed.pt",
    val_tensor="uclchem_grav_val_preprocessed.pt",
    stoichiometric_matrix="stoichiometric_matrix.pt",
)

autoencoder_preprocessing = PreprocessingConfig(
    name="autoencoder",
    input_source="uclchem_grav",
    train_tensor="autoencoder_train_preprocessed.pt",
    val_tensor="autoencoder_val_preprocessed.pt",
    stoichiometric_matrix=None,
)

autoregressive_preprocessing = PreprocessingConfig(
    name="autoregressive",
    input_source="uclchem_grav",
    train_tensor="autoregressive_train_preprocessed.pt",
    val_tensor="autoregressive_val_preprocessed.pt",
    stoichiometric_matrix=None,
    autoencoder_component="autoencoder_uclchem_grav",
)

autoencoder_emulator_uclchem_grav = SurrogateConfig(
    name="ae_emulator_grav",
    description="Autoencoder + latent Emulator surrogate for gravitational collapse",
    components={
        "encoder": "autoencoder_uclchem_grav",
        "emulator": "emulator_uclchem_grav",
        "decoder": "autoencoder_uclchem_grav",
    },
    rollout_steps=100,
    device="cuda",
)


class DatasetName(StrEnum):
    """Defines supported dataset config names."""

    uclchem_grav = "uclchem_grav"
    carbox_grav = "carbox_grav"

    def config(self) -> DatasetConfig:
        """Returns the dataset config for this dataset name."""
        match self:
            case DatasetName.uclchem_grav:
                return uclchem_grav
            case DatasetName.carbox_grav:
                return carbox_grav
            case _:
                raise ValueError(f"{self} not valid datset")


class PreprocessingName(StrEnum):
    """Defines supported preprocessing config names."""

    uclchem_grav = "uclchem_grav"
    autoencoder = "autoencoder"
    autoregressive = "autoregressive"

    def config(self) -> PreprocessingConfig:
        """Returns the preprocessing config for this preprocessing name."""
        match self:
            case PreprocessingName.uclchem_grav:
                return uclchem_grav_preprocessing
            case PreprocessingName.autoencoder:
                return autoencoder_preprocessing
            case PreprocessingName.autoregressive:
                return autoregressive_preprocessing
            case _:
                raise ValueError(f"{self} not valid preprocessing method")


class ComponentName(StrEnum):
    """Defines supported component config names."""

    autoencoder_uclchem_grav = "autoencoder_uclchem_grav"
    emulator_uclchem_grav = "emulator_uclchem_grav"

    def config(self) -> ComponentConfig:
        """Returns the component config for this component name."""
        match self:
            case ComponentName.autoencoder_uclchem_grav:
                return autoencoder_uclchem_grav
            case ComponentName.emulator_uclchem_grav:
                return emulator_uclchem_grav
            case _:
                raise ValueError(f"{self} not valid component")


class SurrogateName(StrEnum):
    """Defines supported surrogate config names."""

    autoencoder_emulator_uclchem_grav = "autoencoder_emulator_uclchem_grav"

    def config(self) -> SurrogateConfig:
        """Returns the surrogate config for this surrogate name."""
        match self:
            case SurrogateName.autoencoder_emulator_uclchem_grav:
                return autoencoder_emulator_uclchem_grav
            case _:
                raise ValueError(f"{self} not valid surrogate model")


def build_training_config(root: Path, component_name: str) -> TrainingRunConfig:
    """Builds the training config dataclass."""
    runtime = RUNTIME
    component = ComponentName(component_name).config()
    dataset = DatasetName(component.dataset).config()
    preprocessing = PreprocessingName(component.preprocessing_method).config()
    return TrainingRunConfig(
        device=runtime.device,
        seed=runtime.seed,
        root=root,
        paths=runtime.paths,
        dataset=dataset,
        preprocessing=preprocessing,
        component=component,
    )


def _resolve_autoencoder(method: PreprocessingConfig) -> AutoencoderConfig | None:
    """Resolves optional autoencoder config for a preprocessing method."""
    if method.autoencoder_component is None:
        return None
    component = ComponentName(method.autoencoder_component).config()
    if isinstance(component, AutoencoderConfig):
        return component
    raise ValueError(f"{method.autoencoder_component} is not an autoencoder")


def _resolve_dataset_for_preprocess(
    source: str, method: PreprocessingConfig, autoencoder: AutoencoderConfig | None
) -> DatasetConfig:
    """Resolves dataset config used by the preprocessing run."""
    if autoencoder is not None:
        if source in DatasetName.__members__:
            source_dataset = DatasetName(source).config()
            if source_dataset.name != autoencoder.dataset:
                raise ValueError(
                    f"{method.autoencoder_component} dataset does not match {source}"
                )
        return DatasetName(autoencoder.dataset).config()
    if method.name == "uclchem_grav":
        return DatasetName(source).config()
    if source in DatasetName.__members__:
        return DatasetName(source).config()
    source_method = PreprocessingName(source).config()
    return DatasetName(source_method.input_source).config()


def _resolve_input_dir(
    root: Path, dataset_name: str, method: PreprocessingConfig
) -> str | None:
    """Resolves the input directory path for chained preprocessing."""
    if method.name == "uclchem_grav":
        return None
    if method.input_source == dataset_name:
        return str(root / RUNTIME.paths.preprocessed_dir / dataset_name)
    return str(
        root / RUNTIME.paths.preprocessed_dir / dataset_name / method.input_source
    )


def build_preprocess_run_config(
    root: Path, source: str, method_name: str
) -> tuple[PreprocessRunConfig, str]:
    """Builds the preprocessing run config dataclass."""
    method = PreprocessingName(method_name).config()
    autoencoder = _resolve_autoencoder(method)
    dataset = _resolve_dataset_for_preprocess(source, method, autoencoder)
    dataset_name = dataset.name
    input_dir = _resolve_input_dir(root, dataset_name, method)
    if autoencoder is not None and autoencoder.dataset != dataset_name:
        raise ValueError(
            f"{method.autoencoder_component} dataset does not match {dataset_name}"
        )
    return (
        PreprocessRunConfig(
            device=RUNTIME.device,
            seed=RUNTIME.seed,
            root=root,
            paths=RUNTIME.paths,
            dataset=dataset,
            method=method,
            input_dir=input_dir,
            autoencoder=autoencoder,
        ),
        dataset_name,
    )
