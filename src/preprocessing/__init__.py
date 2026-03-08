"""Preprocessing module for dataset preparation."""

from typing import Callable, Dict

from .carbox_grav_preprocessing import preprocess_carbox_grav
from .latent_autoregressive_preprocessing import preprocess_latent_autoregressive
from .uclchem_grav_preprocessing import preprocess_uclchem_grav

# Registry of available preprocessors
PREPROCESSORS: Dict[str, Callable[..., None]] = {
    "uclchem_grav": preprocess_uclchem_grav,
    "carbox_grav": preprocess_carbox_grav,
    "latent_autoregressive": preprocess_latent_autoregressive,
}


def get_preprocessor(dataset_name: str) -> Callable[..., None] | None:
    """Get preprocessor function for a dataset."""
    return PREPROCESSORS.get(dataset_name)


def list_available_datasets() -> list[str]:
    """List all available datasets for preprocessing."""
    return list(PREPROCESSORS.keys())


__all__ = [
    "preprocess_uclchem_grav",
    "preprocess_carbox_grav",
    "preprocess_latent_autoregressive",
    "PREPROCESSORS",
    "get_preprocessor",
    "list_available_datasets",
]
