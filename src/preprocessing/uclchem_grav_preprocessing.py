"""Preprocessing script for UCLCHEM gravitational collapse dataset."""

import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import torch


def round_down_sigfigs(value: float, sig_figs: int = 2) -> float:
    """Round down to specified significant figures."""
    if value == 0:
        return 0.0
    # Calculate the order of magnitude
    magnitude = 10 ** math.floor(math.log10(abs(value)))
    # Scale to significant figures, round down, scale back
    scaled = value / magnitude
    rounded = math.floor(scaled * (10 ** (sig_figs - 1))) / (10 ** (sig_figs - 1))
    return rounded * magnitude


def round_up_sigfigs(value: float, sig_figs: int = 2) -> float:
    """Round up to specified significant figures."""
    if value == 0:
        return 0.0
    # Calculate the order of magnitude
    magnitude = 10 ** math.floor(math.log10(abs(value)))
    # Scale to significant figures, round up, scale back
    scaled = value / magnitude
    rounded = math.ceil(scaled * (10 ** (sig_figs - 1))) / (10 ** (sig_figs - 1))
    return rounded * magnitude


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
    """Save dataset metadata to JSON."""
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
    """Save species list to JSON."""
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
    """Build and save stoichiometric matrix to NPY."""
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


def save_train_val_split(
    output_dir: str,
    train_models: list[int],
    val_models: list[int],
    seed: int = 42,
) -> None:
    """Save train/validation split information to JSON."""
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

    Loads preprocessed data from data/uclchem_grav.npy and performs:
    - Train/validation split
    - Saves train and val datasets as .npy files
    - Computes and saves stoichiometric matrix
    - Computes and saves metadata
    - Computes and saves physical parameter ranges

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

    # Load the preprocessed .npy file
    source_file = "data/uclchem_grav.npy"
    print(f"Loading preprocessed data from {source_file}...")
    data_array = np.load(os.path.join(project_root, source_file))
    print(f"Loaded array with shape {data_array.shape}")

    # Load column mapping
    columns_path = "data/uclchem_grav_columns.json"
    with open(os.path.join(project_root, columns_path), "r") as f:
        columns_mapping = json.load(f)

    # Convert to list of column names in order
    num_cols = data_array.shape[1]
    columns = [columns_mapping[str(i)] for i in range(num_cols)]
    print(f"Columns: {columns[:10]}... (showing first 10)")

    # Define column indices
    model_col_idx = columns.index("Model")

    # Physical parameters
    phys_cols = ["Density", "Radfield", "Av", "gasTemp"]
    phys_col_indices = [columns.index(p) for p in phys_cols]

    # Species columns (everything after metadata and physical params)
    metadata_cols = ["Index", "Model", "Time"]
    species = [col for col in columns if col not in metadata_cols + phys_cols]
    print(f"Found {len(species)} species columns")

    # Get unique models for train/val split
    print("Getting unique models for train/val split...")
    model_col = data_array[:, model_col_idx]
    all_models = np.unique(model_col)
    np.random.shuffle(all_models)

    split_ratio = 0.75
    split_idx = int(len(all_models) * split_ratio)
    train_models = all_models[:split_idx]
    val_models = all_models[split_idx:]
    train_models_set = set(train_models.tolist())
    val_models_set = set(val_models.tolist())

    print(f"Train models: {len(train_models)}, Val models: {len(val_models)}")

    # Split data into train and validation sets
    print("Splitting data into train and validation sets...")
    train_mask = np.isin(model_col, list(train_models_set))
    val_mask = np.isin(model_col, list(val_models_set))

    train_data = data_array[train_mask]
    val_data = data_array[val_mask]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Compute physical parameter ranges from training data
    print("Computing physical parameter ranges...")
    phys_ranges = {}
    units = {
        "Density": "H nuclei per cm^3",
        "Radfield": "Habing field",
        "Av": "Magnitudes",
        "gasTemp": "Kelvin",
    }
    for p, idx in zip(phys_cols, phys_col_indices):
        col_min = float(train_data[:, idx].min())
        col_max = float(train_data[:, idx].max())
        phys_ranges[p] = {
            "min": round_down_sigfigs(col_min),
            "max": round_up_sigfigs(col_max),
            "unit": units.get(p, "unknown"),
        }

    with open(os.path.join(output_dir, "physical_parameter_ranges.json"), "w") as f:
        json.dump(phys_ranges, f, indent=2)

    # Save train and validation datasets as .npy files
    print("Saving train and validation datasets...")
    np.save(os.path.join(output_dir, "train.npy"), train_data)
    np.save(os.path.join(output_dir, "val.npy"), val_data)

    # Save preprocessing artifacts
    save_species(output_dir, species)
    save_stoichiometric_matrix(output_dir, species)

    # Save train/val split info
    save_train_val_split(
        output_dir,
        [int(m) for m in train_models],
        [int(m) for m in val_models],
        seed=seed,
    )

    # Save metadata
    save_metadata(
        output_dir,
        dataset_name,
        source_file,
        num_species=len(species),
        num_physical_params=len(phys_cols),
        num_train_samples=len(train_data),
        num_val_samples=len(val_data),
        train_val_split_ratio=split_ratio,
    )

    print(f"Preprocessing complete. Output saved to {output_dir}")
    print(f"  - Species: {len(species)}")
    print(f"  - Physical parameters: {len(phys_cols)}")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print("  - Physical parameter ranges:")
    for param, info in phys_ranges.items():
        print(f"      {param}: [{info['min']:.6e}, {info['max']:.6e}] {info['unit']}")
