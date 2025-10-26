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


def preprocess_dataset(cfg: DictConfig):
    """Preprocess raw UCLCHEM data into cleaned HDF5 format.

    Args:
        cfg: Hydra configuration with preprocessing parameters.
    """
    print("=" * 80)
    print("Preprocessing UCLCHEM Dataset")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Load raw data
    print(f"\nLoading raw data from {cfg.input_path}...")
    if not os.path.exists(cfg.input_path):
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")

    df = pd.read_hdf(cfg.input_path, key=cfg.input_key)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Drop unwanted columns
    if cfg.columns_to_drop:
        print(f"\nDropping columns: {cfg.columns_to_drop}")
        df = df.drop(columns=cfg.columns_to_drop, errors="ignore")
        print(f"Remaining columns: {len(df.columns)}")

    # Filter physical parameters
    if hasattr(cfg, "min_radfield"):
        print(f"\nFiltering Radfield >= {cfg.min_radfield}")
        initial_len = len(df)
        df = df[df["Radfield"] >= cfg.min_radfield]
        print(f"Filtered {initial_len - len(df)} rows, {len(df)} remaining")

    # Split into train/validation
    print(f"\nSplitting data (train_split={cfg.train_split})...")
    np.random.seed(cfg.seed)

    # Get unique trajectory indices
    if "trajectory_index" in df.columns:
        unique_trajectories = df["trajectory_index"].unique()
        np.random.shuffle(unique_trajectories)

        split_idx = int(len(unique_trajectories) * cfg.train_split)
        train_trajectories = unique_trajectories[:split_idx]
        val_trajectories = unique_trajectories[split_idx:]

        train_df = df[df["trajectory_index"].isin(train_trajectories)]
        val_df = df[df["trajectory_index"].isin(val_trajectories)]

        print(f"Train trajectories: {len(train_trajectories)}")
        print(f"Validation trajectories: {len(val_trajectories)}")
    else:
        # Simple random split if no trajectory index
        split_idx = int(len(df) * cfg.train_split)
        indices = np.arange(len(df))
        np.random.shuffle(indices)

        train_df = df.iloc[indices[:split_idx]]
        val_df = df.iloc[indices[split_idx:]]

    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Save to HDF5
    print(f"\nSaving preprocessed data to {cfg.output_path}...")
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

    with pd.HDFStore(cfg.output_path, mode="w") as store:
        store.put("train", train_df, format="table")
        store.put("val", val_df, format="table")

    print("\nPreprocessing complete!")
    print(f"Saved to: {cfg.output_path}")
    print(f"  - /train: {len(train_df)} rows")
    print(f"  - /val: {len(val_df)} rows")


@hydra.main(config_path="../../../configs", config_name="preprocess", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for data preprocessing CLI command.

    Args:
        cfg: Hydra configuration object from preprocess.yaml with CLI overrides.
    """
    # Get working directory
    cfg.working_path = os.getcwd()

    # Run preprocessing
    preprocess_dataset(cfg)


if __name__ == "__main__":
    main()
