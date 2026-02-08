"""Builds runtime config objects from in-code dataclass registries."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.configs.types import (
    AutoencoderConfig,
    ComponentConfig,
    DatasetConfig,
    EmulatorConfig,
    SurrogateConfig,
)

_RUNTIME = SimpleNamespace(
    device="cuda",
    seed=42,
    paths=SimpleNamespace(
        weights_dir="outputs/weights",
        preprocessed_dir="outputs/preprocessed",
    ),
)

_DATASETS: dict[str, DatasetConfig] = {
    "grav": DatasetConfig(
        name="grav",
        raw_path="data/gravitational_collapse.h5",
        input_key="df",
        n_species=333,
        species_file="data/species.txt",
        initial_abundances="data/initial_abundances.npy",
        phys=["Density", "Radfield", "Av", "gasTemp"],
        physical_ranges={
            "Density": (68481.0, 1284211415.0),
            "Radfield": (1.0e-4, 26.0),
            "Av": (0.1, 6914.0),
            "gasTemp": (13.0, 133.0),
        },
        n_params=4,
        metadata_columns=["Index", "Model", "Time"],
        columns_to_drop=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"],
        num_metadata=3,
        num_phys=4,
        abundances_lower=1.0e-20,
        abundances_upper=1.0,
        stoichiometric_matrix_path="outputs/preprocessed/grav/initial/stoichiometric_matrix.pt",
        train_split=0.75,
        seed=42,
        num_species=333,
    ),
    "carbox_grav": DatasetConfig(
        name="carbox_grav",
        raw_path="data/carbox_gravitational_collapse.h5",
        input_key="large",
        n_species=163,
        species_file="data/carbox_grav_species.txt",
        initial_abundances="data/initial_abundances.npy",
        phys=["Density", "Radfield", "Av", "gasTemp"],
        physical_ranges={
            "Density": (0.5715546011924744, 195699.671875),
            "Radfield": (3.8874722463333455e-07, 3.748945713043213),
            "Av": (0.0075920443050563335, 18.559925079345703),
            "gasTemp": (2.591064214706421, 7209.87451171875),
        },
        n_params=4,
        metadata_columns=["Index", "Model", "Time"],
        columns_to_drop=[],
        num_metadata=3,
        num_phys=4,
        abundances_lower=1.0e-20,
        abundances_upper=1.0,
        stoichiometric_matrix_path="outputs/preprocessed/carbox_grav/initial/stoichiometric_matrix.pt",
        train_split=0.75,
        seed=42,
        num_species=163,
    ),
}

_COMPONENTS: dict[str, ComponentConfig] = {
    "autoencoder_grav": AutoencoderConfig(
        name="autoencoder_grav_large",
        type="autoencoder",
        dataset="grav",
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
        latents_minmax_path="outputs/weights/autoencoder_grav_large/latents_minmax.npy",
        power_weight=20.0,
        conservation_weight=100.0,
        pretrained_model_path="outputs/weights/autoencoder_grav_large/weights.pth",
    ),
    "emulator_grav": EmulatorConfig(
        name="emulator_grav_sequential",
        type="emulator",
        dataset="grav",
        preprocessing_method="autoregressive",
        autoencoder_component="autoencoder_grav_large",
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
    ),
}

_COMPONENTS["autoencoder_grav_large"] = _COMPONENTS["autoencoder_grav"]
_COMPONENTS["emulator_grav_sequential"] = _COMPONENTS["emulator_grav"]

_PREPROCESSING: dict[str, SimpleNamespace] = {
    "initial": SimpleNamespace(
        name="initial",
        input_type="dataset",
        input_source="grav",
        train_tensor="initial_train_preprocessed.pt",
        val_tensor="initial_val_preprocessed.pt",
        stoichiometric_matrix="stoichiometric_matrix.pt",
    ),
    "autoencoder": SimpleNamespace(
        name="autoencoder",
        input_type="preprocessing",
        input_source="initial",
        train_tensor="autoencoder_train_preprocessed.pt",
        val_tensor="autoencoder_val_preprocessed.pt",
        stoichiometric_matrix=None,
    ),
    "autoregressive": SimpleNamespace(
        name="autoregressive",
        input_type="preprocessing",
        input_source="initial",
        train_tensor="autoregressive_train_preprocessed.pt",
        val_tensor="autoregressive_val_preprocessed.pt",
        stoichiometric_matrix=None,
    ),
}

_SURROGATES: dict[str, SurrogateConfig] = {
    "ae_emulator_grav": SurrogateConfig(
        name="ae_emulator_grav",
        description="Autoencoder + latent Emulator surrogate for gravitational collapse",
        components={
            "encoder": "autoencoder_grav_large",
            "emulator": "emulator_grav_sequential",
            "decoder": "autoencoder_grav_large",
        },
        rollout_steps=100,
        device="cuda",
    )
}


def load_runtime(_: Path) -> SimpleNamespace:
    """Returns runtime settings from the in-code registry."""
    return _RUNTIME


def load_dataset_config(_: Path, name: str) -> DatasetConfig:
    """Returns one dataset config by name."""
    return _DATASETS[name]


def load_component_config(_: Path, name: str) -> ComponentConfig:
    """Returns one component config by name."""
    return _COMPONENTS[name]


def load_preprocessing_config(_: Path, name: str) -> SimpleNamespace:
    """Returns one preprocessing config by name."""
    return _PREPROCESSING[name]


def _find_autoencoder(dataset_name: str) -> AutoencoderConfig | None:
    """Finds one autoencoder component for a dataset."""
    for component in _COMPONENTS.values():
        if (
            isinstance(component, AutoencoderConfig)
            and component.dataset == dataset_name
        ):
            return component
    return None


def _build_input_dir(
    root: Path, dataset_name: str, source: str, runtime: SimpleNamespace
) -> str:
    """Builds preprocessing input path for chained preprocess steps."""
    return str(root / runtime.paths.preprocessed_dir / dataset_name / source)


def build_training_config(root: Path, component_name: str) -> SimpleNamespace:
    """Builds the training config namespace."""
    runtime = load_runtime(root)
    component = load_component_config(root, component_name)
    dataset = load_dataset_config(root, component.dataset)
    preprocessing = load_preprocessing_config(root, component.preprocessing_method)
    return SimpleNamespace(
        device=runtime.device,
        seed=runtime.seed,
        root=root,
        paths=runtime.paths,
        dataset=dataset,
        preprocessing=preprocessing,
        component=component,
    )


def _resolve_dataset_name(
    root: Path, source: str, method: SimpleNamespace
) -> tuple[str, str | None]:
    """Resolves dataset source name and optional input directory."""
    runtime = load_runtime(root)
    if method.input_type == "dataset":
        return source, None
    source_method = load_preprocessing_config(root, source)
    dataset_name = source_method.input_source
    return dataset_name, _build_input_dir(root, dataset_name, source, runtime)


def build_preprocess_run_config(
    root: Path, source: str, method_name: str
) -> tuple[SimpleNamespace, str]:
    """Builds the preprocessing config namespace."""
    runtime = load_runtime(root)
    method = load_preprocessing_config(root, method_name)
    dataset_name, input_dir = _resolve_dataset_name(root, source, method)
    dataset = load_dataset_config(root, dataset_name)
    autoencoder = (
        _find_autoencoder(dataset_name) if method.name == "autoregressive" else None
    )
    return (
        SimpleNamespace(
            device=runtime.device,
            seed=runtime.seed,
            root=root,
            paths=runtime.paths,
            dataset=dataset,
            method=method,
            input_dir=input_dir,
            autoencoder=autoencoder,
        ),
        dataset_name,
    )


def load_surrogate_config(_: Path, name: str) -> SurrogateConfig:
    """Returns one surrogate config by name."""
    return _SURROGATES[name]


def config_to_dict(config: Any) -> Any:
    """Converts dataclasses and namespaces into plain dictionaries."""
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, SimpleNamespace):
        return {k: config_to_dict(v) for k, v in vars(config).items()}
    if isinstance(config, dict):
        return {k: config_to_dict(v) for k, v in config.items()}
    return config
