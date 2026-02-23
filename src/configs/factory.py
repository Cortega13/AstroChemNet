"""Config builders.

Centralizes dataset-driven configuration selection so AE/EM configs always match
the chosen dataset.
"""

from __future__ import annotations

from typing import Any

from src.configs.autoencoder import AEConfig
from src.configs.datasets import DATASET_SPECS, DatasetConfig, DatasetSpec
from src.configs.emulator import EMConfig


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_name]
    except KeyError as e:
        available = ", ".join(DATASET_SPECS.keys())
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available: {available}"
        ) from e


def _merged_kwargs(
    base: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    if not overrides:
        return dict(base)
    merged = dict(base)
    merged.update(overrides)
    return merged


def build_dataset_config(dataset_name: str) -> DatasetConfig:
    return DatasetConfig(dataset_name=dataset_name)


def build_ae_config(
    dataset_config: DatasetConfig,
    overrides: dict[str, Any] | None = None,
) -> AEConfig:
    spec = get_dataset_spec(dataset_config.dataset_name)
    kwargs = _merged_kwargs(spec.ae_kwargs, overrides)
    return AEConfig(dataset_config=dataset_config, **kwargs)


def build_em_config(
    dataset_config: DatasetConfig,
    ae_config: AEConfig,
    overrides: dict[str, Any] | None = None,
) -> EMConfig:
    spec = get_dataset_spec(dataset_config.dataset_name)
    kwargs = _merged_kwargs(spec.em_kwargs, overrides)
    return EMConfig(dataset_config=dataset_config, ae_config=ae_config, **kwargs)
