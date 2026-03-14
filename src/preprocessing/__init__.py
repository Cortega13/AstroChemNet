"""Preprocessing module for dataset preparation."""

from typing import Callable, Dict

from .autoregressive_preprocessing import preprocess_autoregressive
from .carbox_grav_preprocessing import preprocess_carbox_grav
from .latent_autoregressive_preprocessing import preprocess_latent_autoregressive
from .latent_ode_preprocessing import preprocess_latent_ode
from .uclchem_grav_preprocessing import preprocess_uclchem_grav

# Registry of available preprocessors
PREPROCESSORS: Dict[str, Callable[..., None]] = {
    "uclchem_grav": preprocess_uclchem_grav,
    "carbox_grav": preprocess_carbox_grav,
    "autoregressive": preprocess_autoregressive,
    "latent_autoregressive": preprocess_latent_autoregressive,
    "latent_ode": preprocess_latent_ode,
}


def get_preprocessor(dataset_name: str) -> Callable[..., None] | None:
    """Get preprocessor function for a dataset."""
    return PREPROCESSORS.get(dataset_name)


def list_available_datasets() -> list[str]:
    """List all available datasets for preprocessing."""
    return list(PREPROCESSORS.keys())


__all__ = [
    "preprocess_autoregressive",
    "preprocess_uclchem_grav",
    "preprocess_carbox_grav",
    "preprocess_latent_autoregressive",
    "preprocess_latent_ode",
    "PREPROCESSORS",
    "get_preprocessor",
    "list_available_datasets",
]
