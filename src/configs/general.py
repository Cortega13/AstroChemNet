"""General runtime and dataset configuration."""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch


# NOTE: Not using field() for class attributes - we want simple defaults without dataclass field overhead
@dataclass
class GeneralConfig:
    """General configuration for runtime and dataset information.

    Attributes:
        dataset_name: Name of the dataset to load preprocessing artifacts for
    """

    dataset_name: str = "uclchem_grav"

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
    preprocessing_dir: str = None  # type: ignore
    dataset_path: str = None  # type: ignore
    columns_mapping_path: str = None  # type: ignore
    physical_parameter_ranges: dict[str, tuple[float, float]] = None  # type: ignore
    stoichiometric_matrix: np.ndarray = None  # type: ignore
    species: list[str] = None  # type: ignore
    phys: list[str] = None  # type: ignore
    num_metadata: int = None  # type: ignore
    num_phys: int = None  # type: ignore
    num_species: int = None  # type: ignore

    def __post_init__(self) -> None:
        """Load configuration from preprocessing output after initialization."""
        # Set up paths based on dataset_name
        self.preprocessing_dir = os.path.join(
            self.project_root, "outputs", "preprocessed", self.dataset_name
        )
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
