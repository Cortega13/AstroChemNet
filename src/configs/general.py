"""General runtime and dataset configuration."""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GeneralConfig:
    """General configuration for runtime and dataset information.

    Attributes:
        dataset_name: Name of the dataset to load preprocessing artifacts for
    """

    dataset_name: str = "uclchem_grav"

    # Class-level constants (not affected by dataset_name)
    working_path: str = field(
        default_factory=lambda: os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        ),
        init=False,
        repr=False,
    )
    project_root: str = field(
        default_factory=lambda: os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ),
        init=False,
        repr=False,
    )
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        init=False,
        repr=False,
    )

    # Abundances clipping constants
    abundances_lower_clipping: np.float32 = field(
        default=np.float32(1e-20),
        init=False,
        repr=False,
    )
    abundances_upper_clipping: np.float32 = field(
        default=np.float32(1),
        init=False,
        repr=False,
    )

    # Metadata column names (constant across datasets)
    metadata: list[str] = field(
        default_factory=lambda: ["Index", "Model", "Time"],
        init=False,
        repr=False,
    )

    # Instance attributes loaded from preprocessing output
    preprocessing_dir: str = field(init=False, repr=False)
    dataset_path: str = field(init=False, repr=False)
    physical_parameter_ranges: dict[str, tuple[float, float]] = field(
        init=False, repr=False
    )
    initial_abundances: np.ndarray = field(init=False, repr=False)
    stoichiometric_matrix: np.ndarray = field(init=False, repr=False)
    species: list[str] = field(init=False, repr=False)
    phys: list[str] = field(init=False, repr=False)
    num_metadata: int = field(init=False, repr=False)
    num_phys: int = field(init=False, repr=False)
    num_species: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load configuration from preprocessing output after initialization."""
        # Set up paths based on dataset_name
        self.preprocessing_dir = os.path.join(
            self.project_root, "outputs", "preprocessed", self.dataset_name
        )
        self.dataset_path = os.path.join(
            self.project_root, f"data/{self.dataset_name}_clean.h5"
        )

        # Load physical parameter ranges from JSON
        self.physical_parameter_ranges = self._load_physical_parameter_ranges()

        # Load species from JSON
        self.species = self._load_species()

        # Load initial abundances from NPY
        initial_abundances_path = os.path.join(
            self.preprocessing_dir, "initial_abundances.npy"
        )
        self.initial_abundances = np.load(initial_abundances_path)

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

    def _load_species(self) -> list[str]:
        """Load species list from preprocessing output.

        Returns:
            List of species names
        """
        species_path = os.path.join(self.preprocessing_dir, "species.json")
        with open(species_path, "r") as f:
            species_data = json.load(f)
        return species_data["species"]
