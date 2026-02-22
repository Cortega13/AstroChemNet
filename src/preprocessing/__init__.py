"""Preprocessing module for dataset preparation."""

from typing import Callable, Dict

from .carbox_grav_preprocessing import preprocess_carbox_grav
from .emulator_preprocessing import preprocess_emulator
from .uclchem_grav_preprocessing import preprocess_uclchem_grav

# Registry of available preprocessors
PREPROCESSORS: Dict[str, Callable[..., None]] = {
    "uclchem_grav": preprocess_uclchem_grav,
    "carbox_grav": preprocess_carbox_grav,
    "emulator": preprocess_emulator,
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
    "preprocess_emulator",
    "PREPROCESSORS",
    "get_preprocessor",
    "list_available_datasets",
]
