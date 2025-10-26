"""CLI command for preprocessing raw UCLCHEM data.

This module provides the entry point for preprocessing raw UCLCHEM outputs
into the cleaned HDF5 format used by AstroChemNet training scripts. This
includes filtering columns, splitting into train/validation sets, and
applying basic data cleaning operations.

Usage:
    astrochemnet-preprocess [OPTIONS]

Examples:
    # Preprocess with default config
    astrochemnet-preprocess

    # Override input/output paths
    astrochemnet-preprocess input_path=data/raw.h5 output_path=data/clean.h5

    # Change train/validation split
    astrochemnet-preprocess train_split=0.8

    # Override physical parameter filtering
    astrochemnet-preprocess min_radfield=1.0e-5
"""

import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..utils import rename_columns


def load_initial_abundances(cfg: DictConfig, species: np.ndarray):
    """Loads initial abundances with uninitialized information."""
    initial_abundances = np.load(cfg.dataset.initial_abundances_path)
    df_init = pd.DataFrame(initial_abundances, columns=species)
    df_init["Radfield"] = 0
    df_init["Time"] = 0
    df_init["Av"] = 0
    df_init["gasTemp"] = 0
    df_init["Density"] = 0
    return df_init


def preprocess_dataset(cfg: DictConfig):
    """Preprocess raw UCLCHEM data into cleaned HDF5 format.

    Args:
        cfg: Hydra configuration with dataset parameters.
    """
    print("=" * 80)
    print("Preprocessing UCLCHEM Dataset")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Load species list
    species = np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()

    # Load raw data
    print(f"\nLoading raw data from {cfg.dataset.preprocessing.input_path}...")
    df = pd.read_hdf(
        cfg.dataset.preprocessing.input_path, key=cfg.dataset.preprocessing.input_key
    )
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Rename columns
    df.columns = rename_columns(df.columns)

    # Drop unwanted columns
    if cfg.dataset.preprocessing.columns_to_drop:
        print(f"\nDropping columns: {cfg.dataset.preprocessing.columns_to_drop}")
        df = df.drop(columns=cfg.dataset.preprocessing.columns_to_drop, errors="ignore")
        print(f"Remaining columns: {len(df.columns)}")

    # Set minimum radfield. Radfield goes several orders of magnitude lower than 1e-4, but I suspect this has little effect on abundances.
    if hasattr(cfg.dataset.preprocessing, "min_radfield"):
        print(f"Setting Minimum Radfield to {cfg.dataset.preprocessing.min_radfield}")
        df["Radfield"] = np.maximum(
            df["Radfield"], cfg.dataset.preprocessing.min_radfield
        )

    # Load initial abundances.
    df_init = load_initial_abundances(cfg, species)

    # Setting the first row for each trajectory to initial abundances
    output_chunks = []
    df.sort_values(by=["Model", "Time"], inplace=True)  # type: ignore
    for _, tdf in df.groupby("Model", sort=False):
        tdf = tdf.reset_index(drop=True)

        df_init["Model"] = tdf.iloc[0]["Model"]

        tdf = pd.concat([df_init, tdf], ignore_index=True)

        physical = tdf[cfg.dataset.phys].shift(-1)
        physical.iloc[-1] = physical.iloc[-2]

        tdf.loc[:, cfg.dataset.phys] = physical.values
        output_chunks.append(tdf)

    # Combine all processed chunks and sort columns.
    df = pd.concat(output_chunks, ignore_index=True)
    df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
    df.insert(0, "Index", range(len(df)))
    df = df[cfg.dataset.metadata + cfg.dataset.phys + species]

    # Apply species lower and upper clipping.
    df[species] = df[species].clip(
        lower=cfg.dataset.abundances_clipping.lower,
        upper=cfg.dataset.abundances_clipping.upper,
    )
    # Split into train/validation
    print(f"\nSplitting data (train_split={cfg.dataset.preprocessing.train_split})...")
    np.random.seed(cfg.dataset.preprocessing.seed)

    # Shuffling the tracer indices.
    tracers = df["Model"].unique()
    np.random.shuffle(tracers)

    # Train/validation split based off the shuffled tracer indices.
    split_idx = int(len(tracers) * cfg.dataset.preprocessing.train_split)

    train_tracers = tracers[:split_idx]
    val_tracers = tracers[split_idx:]

    train_df = df[df["Model"].isin(train_tracers)]
    val_df = df[df["Model"].isin(val_tracers)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_hdf(cfg.dataset.preprocessing.output_path, key="train", mode="w")
    val_df.to_hdf(cfg.dataset.preprocessing.output_path, key="val", mode="a")

    print("\nPreprocessing complete!")
    print(f"Saved to: {cfg.dataset.preprocessing.output_path}")
    print(f"  - /train: {len(train_df)} rows")
    print(f"  - /val: {len(val_df)} rows")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for data preprocessing CLI command.

    Args:
        cfg: Hydra configuration object from preprocess.yaml with CLI overrides.
    """
    # Run preprocessing
    preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
