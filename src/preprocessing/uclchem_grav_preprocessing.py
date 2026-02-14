"""Preprocessing script for UCLCHEM gravitational collapse dataset."""

import json
import os
from datetime import datetime, timezone
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch

from .. import utils

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def get_output_dir(project_root: str, dataset_name: str) -> str:
    """Get output directory path for a dataset.

    Args:
        project_root: Path to the project root directory
        dataset_name: Name of the dataset

    Returns:
        Path to the preprocessing output directory
    """
    return os.path.join(project_root, "outputs", "preprocessed", dataset_name)


def save_metadata(
    output_dir: str,
    dataset_name: str,
    source_file: str,
    num_species: int,
    num_physical_params: int,
    num_train_samples: int,
    num_val_samples: int,
    train_val_split_ratio: float = 0.75,
) -> None:
    """Save dataset metadata to JSON.

    Args:
        output_dir: Path to the preprocessing output directory
        dataset_name: Name of the dataset
        source_file: Path to the source data file
        num_species: Number of species in the dataset
        num_physical_params: Number of physical parameters
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        train_val_split_ratio: Ratio of train/validation split
    """
    metadata = {
        "dataset_name": dataset_name,
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source_file,
        "num_species": num_species,
        "num_physical_params": num_physical_params,
        "num_train_samples": num_train_samples,
        "num_val_samples": num_val_samples,
        "train_val_split_ratio": train_val_split_ratio,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def save_species(
    output_dir: str,
    species: list[str],
    elements: list[str] | None = None,
) -> None:
    """Save species list to JSON.

    Args:
        output_dir: Path to the preprocessing output directory
        species: List of species names
        elements: List of elements (optional)
    """
    if elements is None:
        elements = ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]

    species_data = {
        "species": species,
        "num_species": len(species),
        "elements": elements,
    }
    with open(os.path.join(output_dir, "species.json"), "w") as f:
        json.dump(species_data, f, indent=2)


def save_stoichiometric_matrix(
    output_dir: str,
    species: list[str],
    elements: list[str] | None = None,
) -> np.ndarray:
    """Build and save stoichiometric matrix to NPY.

    The stoichiometric matrix S is built such that x @ S gives elemental abundances.

    Args:
        output_dir: Path to the preprocessing output directory
        species: List of species names
        elements: List of elements (optional)

    Returns:
        The stoichiometric matrix (transposed for use with x @ S)
    """
    import re

    if elements is None:
        elements = ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]

    stoichiometric_matrix = np.zeros((len(elements), len(species)))
    modified_species = [s.replace("@", "").replace("#", "") for s in species]

    elements_patterns = {
        "H": re.compile(r"H(?!E)(\d*)"),
        "HE": re.compile(r"HE(\d*)"),
        "C": re.compile(r"C(?!L)(\d*)"),
        "N": re.compile(r"N(\d*)"),
        "O": re.compile(r"O(\d*)"),
        "S": re.compile(r"S(?!I)(\d*)"),
        "SI": re.compile(r"SI(\d*)"),
        "MG": re.compile(r"MG(\d*)"),
        "CL": re.compile(r"CL(\d*)"),
    }

    for element, pattern in elements_patterns.items():
        elem_index = elements.index(element)
        for i, sp in enumerate(modified_species):
            match = pattern.search(sp)
            if match and sp not in ["SURFACE", "BULK"]:
                multiplier = int(match.group(1)) if match.group(1) else 1
                stoichiometric_matrix[elem_index, i] = multiplier

    stoichiometric_matrix_t = stoichiometric_matrix.T
    np.save(
        os.path.join(output_dir, "stoichiometric_matrix.npy"), stoichiometric_matrix_t
    )
    return stoichiometric_matrix_t


def save_physical_parameter_ranges(
    output_dir: str,
    df: pd.DataFrame,
    physical_params: list[str],
) -> dict[str, dict[str, float | str]]:
    """Compute and save min/max values for physical parameters to JSON.

    Args:
        output_dir: Path to the preprocessing output directory
        df: DataFrame containing the data
        physical_params: List of physical parameter column names

    Returns:
        Dictionary of parameter ranges with min, max, and unit
    """
    units = {
        "Density": "H nuclei per cm^3",
        "Radfield": "Habing field",
        "Av": "Magnitudes",
        "gasTemp": "Kelvin",
    }

    ranges = {}
    for param in physical_params:
        ranges[param] = {
            "min": float(df[param].min()),
            "max": float(df[param].max()),
            "unit": units.get(param, "unknown"),
        }

    with open(os.path.join(output_dir, "physical_parameter_ranges.json"), "w") as f:
        json.dump(ranges, f, indent=2)

    return ranges


def save_initial_abundances(
    output_dir: str,
    initial_abundances: np.ndarray,
) -> None:
    """Save initial abundances to NPY.

    Args:
        output_dir: Path to the preprocessing output directory
        initial_abundances: Initial abundances array
    """
    np.save(os.path.join(output_dir, "initial_abundances.npy"), initial_abundances)


def save_train_val_split(
    output_dir: str,
    train_models: list[str],
    val_models: list[str],
    seed: int = 42,
) -> None:
    """Save train/validation split information to JSON.

    Args:
        output_dir: Path to the preprocessing output directory
        train_models: List of model names in training set
        val_models: List of model names in validation set
        seed: Random seed used for splitting
    """
    split_info = {
        "train_models": train_models,
        "val_models": val_models,
        "seed": seed,
    }
    with open(os.path.join(output_dir, "train_val_split.json"), "w") as f:
        json.dump(split_info, f, indent=2)


def preprocess_uclchem_grav(
    dataset_name: str = "uclchem_grav",
    force: bool = False,
) -> None:
    """Preprocess UCLCHEM gravitational collapse dataset.

    Reads raw data from data/uclchem_grav.h5 and writes:
    - Cleaned training and validation sets to data/uclchem_grav_clean.h5
    - Preprocessing artifacts to outputs/preprocessed/<dataset_name>/

    Args:
        dataset_name: Name for the output directory
        force: If True, overwrite existing preprocessing output
    """
    # Determine paths
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_dir = get_output_dir(project_root, dataset_name)

    # Check if output already exists
    if os.path.exists(output_dir) and not force:
        print(f"Preprocessing output already exists at {output_dir}")
        print("Use --force to overwrite")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load initial abundances from old location (will be migrated)
    old_utils_path = os.path.join(project_root, "outputs", "utils")
    initial_abundances = np.load(os.path.join(old_utils_path, "initial_abundances.npy"))

    # Load old species list for initial abundances
    old_species = np.loadtxt(
        os.path.join(old_utils_path, "species.txt"),
        dtype=str,
        delimiter=" ",
        comments=None,
    ).tolist()

    df_inits = pd.DataFrame(initial_abundances, columns=old_species)
    df_inits["Radfield"] = 0
    df_inits["Time"] = 0
    df_inits["Av"] = 0
    df_inits["gasTemp"] = 0
    df_inits["Density"] = 0

    # Load raw data
    source_file = "data/uclchem_grav.h5"
    df = pd.read_hdf(os.path.join(project_root, source_file), key="df", start=0)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input dataset is not a dataframe.")

    # Drop unused columns
    df = df.drop(columns=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"])
    df.columns = utils.rename_columns(list(df.columns))

    # Define metadata and physical parameter columns
    metadata_cols = ["Index", "Model", "Time"]
    phys_cols = ["Density", "Radfield", "Av", "gasTemp"]

    # Extract species columns
    species = [col for col in df.columns if col not in metadata_cols + phys_cols]
    species = sorted(species)

    df.sort_values(by=["Model", "Time"], inplace=True)

    # Clamp Radfield to minimum value
    df["Radfield"] = np.maximum(df["Radfield"], 1e-4)

    output_chunks = []
    params = phys_cols

    for tracer, tdf in df.groupby("Model", sort=False):
        tdf = tdf.reset_index(drop=True)

        df_inits["Model"] = tdf.iloc[0]["Model"]

        tdf = pd.concat([df_inits, tdf], ignore_index=True)

        physical = tdf[params].shift(-1)
        physical.iloc[-1] = physical.iloc[-2]

        tdf[params] = physical

        output_chunks.append(tdf)

    df = pd.concat(output_chunks, ignore_index=True)
    df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
    df.insert(0, "Index", range(len(df)))

    df = df[metadata_cols + params + species]

    # Compute physical parameter ranges BEFORE any transformation
    physical_param_ranges = save_physical_parameter_ranges(output_dir, df, params)

    # Train/validation split
    tracers = df["Model"].unique()
    np.random.shuffle(tracers)

    split_ratio = 0.75
    split_idx = int(len(tracers) * split_ratio)

    train_tracers = tracers[:split_idx]
    val_tracers = tracers[split_idx:]

    train_df = df[df["Model"].isin(train_tracers)]
    val_df = df[df["Model"].isin(val_tracers)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Save cleaned data
    train_df.to_hdf(
        os.path.join(project_root, "data/uclchem_grav_clean.h5"), key="train", mode="w"
    )
    val_df.to_hdf(
        os.path.join(project_root, "data/uclchem_grav_clean.h5"), key="val", mode="a"
    )

    # Save preprocessing artifacts
    save_species(output_dir, species)
    save_stoichiometric_matrix(output_dir, species)
    save_initial_abundances(output_dir, initial_abundances)
    save_train_val_split(output_dir, list(train_tracers), list(val_tracers), seed=seed)
    save_metadata(
        output_dir,
        dataset_name,
        source_file,
        num_species=len(species),
        num_physical_params=len(params),
        num_train_samples=len(train_df),
        num_val_samples=len(val_df),
        train_val_split_ratio=split_ratio,
    )

    print(f"Preprocessing complete. Output saved to {output_dir}")
    print(f"  - Species: {len(species)}")
    print(f"  - Physical parameters: {len(params)}")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Validation samples: {len(val_df)}")
    print("  - Physical parameter ranges:")
    for param, info in physical_param_ranges.items():
        print(f"      {param}: [{info['min']:.6e}, {info['max']:.6e}] {info['unit']}")


# Registry of available preprocessors
PREPROCESSORS: Dict[str, Callable[..., None]] = {
    "uclchem_grav": preprocess_uclchem_grav,
}


def get_preprocessor(dataset_name: str) -> Callable[..., None] | None:
    """Get preprocessor function for a dataset.

    Args:
        dataset_name: Name of the dataset to preprocess

    Returns:
        Preprocessor function if found, None otherwise
    """
    return PREPROCESSORS.get(dataset_name)


def list_available_datasets() -> list[str]:
    """List all available datasets for preprocessing.

    Returns:
        List of dataset names
    """
    return list(PREPROCESSORS.keys())
