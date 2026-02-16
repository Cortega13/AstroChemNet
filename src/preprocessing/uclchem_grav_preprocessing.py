"""Preprocessing script for UCLCHEM gravitational collapse dataset."""

import gc
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from .. import utils


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


def save_physical_parameter_ranges(
    output_dir: str,
    df: pd.DataFrame,
    physical_params: list[str],
) -> dict[str, dict[str, float | str]]:
    """Compute and save min/max values for physical parameters to JSON."""
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
    """Save initial abundances to NPY."""
    np.save(os.path.join(output_dir, "initial_abundances.npy"), initial_abundances)


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

    Reads raw data from data/uclchem_grav.h5 and writes:
    - Cleaned training and validation sets to data/uclchem_grav_clean.h5
    - Preprocessing artifacts to outputs/preprocessed/<dataset_name>/

    Memory-optimized: processes data in chunks and writes incrementally.

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

    # Load initial abundances from old location
    initial_abundances = np.load(os.path.join(output_dir, "initial_abundances.npy"))

    # Load species list for initial abundances
    with open(os.path.join(output_dir, "species.json"), "r") as f:
        species_data = json.load(f)
    old_species = species_data["species"]

    df_inits = pd.DataFrame(initial_abundances, columns=old_species)
    df_inits["Radfield"] = 0
    df_inits["Time"] = 0
    df_inits["Av"] = 0
    df_inits["gasTemp"] = 0
    df_inits["Density"] = 0

    # Load raw data
    source_file = "data/uclchem_grav.h5"
    print(f"Loading raw data from {source_file}...")
    df = pd.read_hdf(os.path.join(project_root, source_file), key="df", start=0)
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input dataset is not a dataframe.")

    print(f"Loaded {len(df)} rows")

    # Drop unused columns
    df = df.drop(columns=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"])
    df.columns = utils.rename_columns(list(df.columns))

    # Define metadata and physical parameter columns
    metadata_cols_no_index = ["Model", "Time"]  # Index added separately
    phys_cols = ["Density", "Radfield", "Av", "gasTemp"]

    # Extract species columns (exclude Index, Model, Time, and phys cols)
    species = [
        col for col in df.columns if col not in ["Index", "Model", "Time"] + phys_cols
    ]
    species = sorted(species)

    print("Sorting data by Model and Time...")
    df.sort_values(by=["Model", "Time"], inplace=True)

    # Clamp Radfield to minimum value
    df["Radfield"] = np.maximum(df["Radfield"], 1e-4)

    params = phys_cols

    # Get unique models for splitting BEFORE processing
    print("Getting unique models for train/val split...")
    all_models = df["Model"].unique()
    np.random.shuffle(all_models)

    split_ratio = 0.75
    split_idx = int(len(all_models) * split_ratio)
    train_models_set = set(all_models[:split_idx])
    val_models_set = set(all_models[split_idx:])

    # Process data in chunks by model to save memory
    print("Processing models...")
    output_path = os.path.join(project_root, "data/uclchem_grav_clean.h5")

    # Initialize physical parameter ranges tracking
    phys_ranges = {p: {"min": float("inf"), "max": float("-inf")} for p in params}

    train_rows = 0
    val_rows = 0
    first_write = True
    global_index = 0  # Track global index across all chunks

    # Batch accumulation for efficient writes
    batch_size = 500
    train_batch: list[pd.DataFrame] = []
    val_batch: list[pd.DataFrame] = []

    # Group by model and process
    for i, (tracer, tdf) in enumerate(df.groupby("Model", sort=False)):
        if i % 500 == 0:
            print(f"  Processing model {i}...")

        tdf = tdf.reset_index(drop=True)

        df_inits["Model"] = tdf.iloc[0]["Model"]

        tdf = pd.concat([df_inits, tdf], ignore_index=True)

        physical = tdf[params].shift(-1)
        physical.iloc[-1] = physical.iloc[-2]

        tdf[params] = physical

        # Update physical parameter ranges
        for p in params:
            phys_ranges[p]["min"] = min(phys_ranges[p]["min"], float(tdf[p].min()))
            phys_ranges[p]["max"] = max(phys_ranges[p]["max"], float(tdf[p].max()))

        # Determine if train or val
        is_train = tracer in train_models_set

        # Select columns in correct order (without Index - added below)
        tdf = tdf[metadata_cols_no_index + params + species]

        # Assign global index and defragment
        chunk_len = len(tdf)
        index_col = pd.Series(
            range(global_index, global_index + chunk_len), name="Index"
        )
        tdf = pd.concat([index_col, tdf], axis=1).copy()  # .copy() defragments
        global_index += chunk_len

        if is_train:
            train_rows += chunk_len
            train_batch.append(tdf)
        else:
            val_rows += chunk_len
            val_batch.append(tdf)

        # Write batches when they reach batch_size
        if len(train_batch) >= batch_size:
            batch_df = pd.concat(train_batch, ignore_index=True)
            if first_write:
                batch_df.to_hdf(output_path, key="train", mode="w", append=True)
                first_write = False
            else:
                batch_df.to_hdf(output_path, key="train", mode="a", append=True)
            train_batch.clear()
            gc.collect()

        if len(val_batch) >= batch_size:
            batch_df = pd.concat(val_batch, ignore_index=True)
            if first_write:
                batch_df.to_hdf(output_path, key="val", mode="w", append=True)
                first_write = False
            else:
                batch_df.to_hdf(output_path, key="val", mode="a", append=True)
            val_batch.clear()
            gc.collect()

        # Clean up
        del tdf, physical

    # Write remaining batches
    if train_batch:
        batch_df = pd.concat(train_batch, ignore_index=True)
        if first_write:
            batch_df.to_hdf(output_path, key="train", mode="w", append=True)
            first_write = False
        else:
            batch_df.to_hdf(output_path, key="train", mode="a", append=True)
        train_batch.clear()

    if val_batch:
        batch_df = pd.concat(val_batch, ignore_index=True)
        if first_write:
            batch_df.to_hdf(output_path, key="val", mode="w", append=True)
        else:
            batch_df.to_hdf(output_path, key="val", mode="a", append=True)
        val_batch.clear()

    # Clean up original dataframe
    del df
    gc.collect()

    print(f"Train rows: {train_rows}, Val rows: {val_rows}")

    # Save physical parameter ranges (with rounded min/max for breathing room)
    units = {
        "Density": "H nuclei per cm^3",
        "Radfield": "Habing field",
        "Av": "Magnitudes",
        "gasTemp": "Kelvin",
    }
    physical_param_ranges = {}
    for p in params:
        physical_param_ranges[p] = {
            "min": round_down_sigfigs(phys_ranges[p]["min"]),
            "max": round_up_sigfigs(phys_ranges[p]["max"]),
            "unit": units.get(p, "unknown"),
        }
    with open(os.path.join(output_dir, "physical_parameter_ranges.json"), "w") as f:
        json.dump(physical_param_ranges, f, indent=2)

    # Save preprocessing artifacts
    save_species(output_dir, species)
    save_stoichiometric_matrix(output_dir, species)
    save_initial_abundances(output_dir, initial_abundances)
    # Convert numpy int64 to Python int for JSON serialization
    save_train_val_split(
        output_dir,
        [int(m) for m in train_models_set],
        [int(m) for m in val_models_set],
        seed=seed,
    )
    save_metadata(
        output_dir,
        dataset_name,
        source_file,
        num_species=len(species),
        num_physical_params=len(params),
        num_train_samples=train_rows,
        num_val_samples=val_rows,
        train_val_split_ratio=split_ratio,
    )

    print(f"Preprocessing complete. Output saved to {output_dir}")
    print(f"  - Species: {len(species)}")
    print(f"  - Physical parameters: {len(params)}")
    print(f"  - Training samples: {train_rows}")
    print(f"  - Validation samples: {val_rows}")
    print("  - Physical parameter ranges:")
    for param, info in physical_param_ranges.items():
        print(f"      {param}: [{info['min']:.6e}, {info['max']:.6e}] {info['unit']}")
