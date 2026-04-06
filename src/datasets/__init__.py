"""Dataset config and preprocessing API."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import torch

from .carbox_grav import preprocess_carbox_grav
from .uclchem_grav import preprocess_uclchem_grav


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""

    dataset_name: str
    working_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    abundances_lower_clipping: np.float32 = np.float32(1e-20)
    abundances_upper_clipping: np.float32 = np.float32(1)
    metadata: list[str] = field(default_factory=lambda: ["Index", "Model", "Time"])
    outputs_dir: str = field(init=False)
    dataset_artifacts_dir: str = field(init=False)
    models_dir: str = field(init=False)
    columns_mapping_path: str = field(init=False)
    physical_parameter_ranges: dict[str, tuple[float, float]] = field(init=False)
    stoichiometric_matrix: np.ndarray = field(init=False)
    species: list[str] = field(init=False)
    phys: list[str] = field(init=False)
    num_metadata: int = field(init=False)
    num_phys: int = field(init=False)
    num_species: int = field(init=False)

    def __post_init__(self) -> None:
        """Load configuration from dataset output."""
        self.outputs_dir = os.path.join(self.project_root, "outputs", self.dataset_name)
        self.dataset_artifacts_dir = os.path.join(self.outputs_dir, "dataset")
        self.models_dir = os.path.join(self.outputs_dir, "models")
        columns_in_output = os.path.join(self.dataset_artifacts_dir, "columns.json")
        self.columns_mapping_path = (
            columns_in_output
            if os.path.exists(columns_in_output)
            else os.path.join(
                self.project_root, f"data/{self.dataset_name}_columns.json"
            )
        )
        self.physical_parameter_ranges = self._load_physical_parameter_ranges()
        species_path = os.path.join(self.dataset_artifacts_dir, "species.json")
        with open(species_path) as f:
            species_data = json.load(f)
        self.species = species_data["species"]
        stoichiometric_matrix_path = os.path.join(
            self.dataset_artifacts_dir, "stoichiometric_matrix.npy"
        )
        self.stoichiometric_matrix = np.load(stoichiometric_matrix_path)
        self.phys = list(self.physical_parameter_ranges.keys())
        self.num_metadata = len(self.metadata)
        self.num_phys = len(self.phys)
        self.num_species = len(self.species)

    def model_dir(self, model_name: str) -> str:
        """Return the directory for one model."""
        return os.path.join(self.models_dir, model_name)

    def model_path(self, model_name: str, filename: str) -> str:
        """Return the path for one model artifact."""
        return os.path.join(self.model_dir(model_name), filename)

    def _load_physical_parameter_ranges(self) -> dict[str, tuple[float, float]]:
        """Load physical parameter ranges from preprocessing output."""
        ranges_path = os.path.join(
            self.dataset_artifacts_dir, "physical_parameter_ranges.json"
        )
        with open(ranges_path) as f:
            ranges_data = json.load(f)
        return {
            param: (info["min"], info["max"]) for param, info in ranges_data.items()
        }


class DatasetName(StrEnum):
    """Dataset names for registry lookup."""

    UCLCHEM_GRAV = "uclchem_grav"
    CARBOX_GRAV = "carbox_grav"


AVAILABLE_DATASETS: list[DatasetName] = list(DatasetName)
PREPROCESSORS: dict[DatasetName, Callable[..., None]] = {
    DatasetName.UCLCHEM_GRAV: preprocess_uclchem_grav,
    DatasetName.CARBOX_GRAV: preprocess_carbox_grav,
}


def build_dataset_config(dataset_name: str | DatasetName) -> DatasetConfig:
    """Build dataset-scoped runtime configuration."""
    dataset_name = DatasetName(dataset_name)
    if dataset_name == DatasetName.UCLCHEM_GRAV:
        return DatasetConfig(
            dataset_name=dataset_name,
            abundances_lower_clipping=np.float32(1e-16),
        )
    if dataset_name == DatasetName.CARBOX_GRAV:
        return DatasetConfig(dataset_name=dataset_name)
    available = ", ".join(AVAILABLE_DATASETS)
    raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {available}")


def get_preprocessor(dataset_name: str | DatasetName) -> Callable[..., None] | None:
    """Get preprocessor function for a dataset."""
    return PREPROCESSORS.get(DatasetName(dataset_name))


def list_available_datasets() -> list[str]:
    """List all available datasets for preprocessing."""
    return [dataset.value for dataset in DatasetName]


__all__ = [
    "AVAILABLE_DATASETS",
    "DatasetName",
    "DatasetConfig",
    "PREPROCESSORS",
    "build_dataset_config",
    "get_preprocessor",
    "list_available_datasets",
    "preprocess_carbox_grav",
    "preprocess_uclchem_grav",
]
