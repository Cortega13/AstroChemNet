"""Dataset-scoped runtime + preprocessing-artifact configuration."""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset.

    This object is responsible for:
    - selecting the dataset (via `dataset_name`)
    - locating preprocessing artifacts under `outputs/preprocessed/<dataset_name>`
    - exposing derived dataset properties (species/phys and their counts)
    """

    dataset_name: str

    # Class-level constants (not affected by dataset_name)
    working_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root: str = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Abundances clipping constants
    abundances_lower_clipping: np.float32 = np.float32(1e-20)
    abundances_upper_clipping: np.float32 = np.float32(1)

    # Metadata column names (constant across datasets)
    metadata: list[str] = field(default_factory=lambda: ["Index", "Model", "Time"])

    # Instance attributes loaded from preprocessing output
    preprocessing_dir: str = field(init=False)
    dataset_path: str = field(init=False)
    columns_mapping_path: str = field(init=False)
    weights_dir: str = field(init=False)
    physical_parameter_ranges: dict[str, tuple[float, float]] = field(init=False)
    stoichiometric_matrix: np.ndarray = field(init=False)
    species: list[str] = field(init=False)
    phys: list[str] = field(init=False)
    num_metadata: int = field(init=False)
    num_phys: int = field(init=False)
    num_species: int = field(init=False)

    def __post_init__(self) -> None:
        """Load configuration from preprocessing output after initialization."""
        # Set up paths based on dataset_name
        self.preprocessing_dir = os.path.join(
            self.project_root, "outputs", "preprocessed", self.dataset_name
        )

        self.weights_dir = os.path.join(
            self.project_root, "outputs", "weights", self.dataset_name
        )
        os.makedirs(self.weights_dir, exist_ok=True)
        # Dataset is now stored in the preprocessing output directory as .npy files
        self.dataset_path = self.preprocessing_dir
        # Path to columns mapping JSON (prefer mapping written by the preprocessor)
        columns_in_output = os.path.join(self.preprocessing_dir, "columns.json")
        self.columns_mapping_path = (
            columns_in_output
            if os.path.exists(columns_in_output)
            else os.path.join(
                self.project_root, f"data/{self.dataset_name}_columns.json"
            )
        )

        # Load physical parameter ranges from JSON
        self.physical_parameter_ranges = self._load_physical_parameter_ranges()

        # Load species from JSON
        species_path = os.path.join(self.preprocessing_dir, "species.json")
        with open(species_path, "r") as f:
            species_data = json.load(f)
        self.species = species_data["species"]

        # Load stoichiometric matrix from NPY
        stoichiometric_matrix_path = os.path.join(
            self.preprocessing_dir, "stoichiometric_matrix.npy"
        )
        self.stoichiometric_matrix = np.load(stoichiometric_matrix_path)

        # Set derived attributes
        self.phys = list(self.physical_parameter_ranges.keys())
        self.num_metadata = len(self.metadata)
        self.num_phys = len(self.phys)
        self.num_species = len(self.species)

    def _load_physical_parameter_ranges(self) -> dict[str, tuple[float, float]]:
        """Load physical parameter ranges from preprocessing output.

        Returns:
            Dictionary mapping parameter names to (min, max) tuples
        """
        ranges_path = os.path.join(
            self.preprocessing_dir, "physical_parameter_ranges.json"
        )
        with open(ranges_path, "r") as f:
            ranges_data = json.load(f)

        return {
            param: (info["min"], info["max"]) for param, info in ranges_data.items()
        }


@dataclass(frozen=True)
class DatasetPreset:
    """Dataset-scoped configuration preset.

    Dataset selection (e.g. `carbox_grav`) should be the single runtime switch
    that determines which overrides apply.

    Notes:
        Keep *instances* of :class:`~src.configs.datasets.DatasetConfig` out of
        the module-level registry to avoid import-time file I/O.
    """

    # Kwargs applied when building DatasetConfig(dataset_name=..., **dataset_kwargs)
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    # Kwargs applied when building AEConfig(dataset_config=..., **ae_kwargs)
    ae_kwargs: dict[str, Any] = field(default_factory=dict)

    # Kwargs applied when building ARConfig(dataset_config=..., ae_config=..., **ar_kwargs)
    em_kwargs: dict[str, Any] = field(default_factory=dict)

    # Kwargs applied when building LatentODEConfig(dataset_config=..., ae_config=..., **kwargs)
    latent_ode_kwargs: dict[str, Any] = field(default_factory=dict)

    # Kwargs applied when building AutoregressiveConfig(dataset_config=..., **kwargs)
    autoregressive_kwargs: dict[str, Any] = field(default_factory=dict)


# Central registry used by src/configs/factory.py builders.
DATASET_PRESETS: dict[str, DatasetPreset] = {
    "uclchem_grav": DatasetPreset(
        dataset_kwargs={"abundances_lower_clipping": np.float32(1e-16)}
    ),
    "carbox_grav": DatasetPreset(),
}


AVAILABLE_DATASETS: list[str] = list(DATASET_PRESETS.keys())
