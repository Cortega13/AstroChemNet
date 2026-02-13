"""General runtime and dataset configuration."""

import os

import numpy as np
import torch


class GeneralConfig:
    """General configuration for runtime and dataset information."""

    working_path = os.path.dirname(
        os.path.dirname((os.path.abspath(__file__)))
    )  # The path to the src folder of the project.
    project_root = os.path.dirname(working_path)  # The path to the project root.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(project_root, "data/uclchem_grav.h5")

    physical_parameter_ranges: dict[str, tuple[float, float]] = {
        "Density": (68481, 1284211415),  # H nuclei per cm^3.
        "Radfield": (1e-4, 26),  # Habing field.
        "Av": (1e-1, 6914),  # Magnitudes.
        "gasTemp": (13, 133),  # Kelvin.
    }

    # Abundances are arbitrarily clipped to 1e-20 since anything lower is insignificant.
    abundances_lower_clipping = np.float32(1e-20)

    # All abundances are relative to number of Hydrogen nuclei. Maximum abundance is all hydrogen in elemental form.
    abundances_upper_clipping = np.float32(1)

    initial_abundances_path = os.path.join(
        project_root, "outputs/utils/initial_abundances.npy"
    )
    initial_abundances = np.load(initial_abundances_path)

    stoichiometric_matrix_path = os.path.join(
        project_root, "outputs/utils/stoichiometric_matrix.npy"
    )
    stoichiometric_matrix = np.load(stoichiometric_matrix_path)

    metadata = ["Index", "Model", "Time"]
    phys = list(physical_parameter_ranges.keys())
    species_path = os.path.join(project_root, "outputs/utils/species.txt")
    species = np.loadtxt(species_path, dtype=str, delimiter=" ", comments=None).tolist()

    num_metadata = len(metadata)
    num_phys = len(phys)
    num_species = len(species)
