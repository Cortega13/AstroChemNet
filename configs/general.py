import os
import torch
import numpy as np


class GeneralConfig:
    working_path = os.path.dirname(
        os.path.dirname((os.path.abspath(__file__)))
    )  # The path to the root folder of the project.
    device = torch.device("cpu")
    dataset_path = os.path.join(working_path, "data/grav_collapse_clean.h5")

    num_timesteps_per_model = 296  # Duration that each model runs for. Multiply by timestep_duration to get total evolution time.
    timestep_duration = 92.9  # In years
    physical_parameter_ranges = {
        "Density": (68481, 1284211415),  # H nuclei per cm^3.
        "Radfield": (1e-4, 26),  # Habing field.
        "Av": (1e-1, 6914),  # Magnitudes.
        "gasTemp": (13, 133),  # Kelvin.
    }
    abundances_lower_clipping = np.float32(
        1e-20
    )  # Abundances are arbitrarily clipped to 1e-20 since anything lower is insignificant.
    abundances_upper_clipping = np.float32(
        1
    )  # All abundances are relative to number of Hydrogen nuclei. Maximum abundance is all hydrogen in elemental form.

    initial_abundances_path = os.path.join(working_path, "utils/initial_abundances.npy")
    initial_abundances = np.load(initial_abundances_path)

    stoichiometric_matrix_path = os.path.join(
        working_path, "utils/stoichiometric_matrix.npy"
    )
    stoichiometric_matrix = np.load(stoichiometric_matrix_path)

    metadata = ["Index", "Model", "Time"]
    phys = list(physical_parameter_ranges.keys())
    species_path = os.path.join(working_path, "utils/species.txt")
    species = np.loadtxt(species_path, dtype=str, delimiter=" ", comments=None).tolist()

    num_metadata = len(metadata)
    num_phys = len(phys)
    num_species = len(species)
