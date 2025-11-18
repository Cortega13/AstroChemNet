"""Initial preprocessor for raw UCLCHEM data."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..utils import rename_columns


class InitialPreprocessor:
    def __init__(self, dataset_cfg: DictConfig, method_cfg: DictConfig, root: Path):
        self.dataset_cfg = dataset_cfg
        self.method_cfg = method_cfg
        self.root = root

    def load_initial_abundances(self, species: list):
        """Loads initial abundances with uninitialized information."""
        initial_abundances = np.load(self.dataset_cfg.initial_abundances_path)
        df_init = pd.DataFrame(initial_abundances, columns=species)
        df_init["Radfield"] = 0
        df_init["Time"] = 0
        df_init["Av"] = 0
        df_init["gasTemp"] = 0
        df_init["Density"] = 0
        return df_init

    def run(self, output_dir: Path):
        """Preprocess raw UCLCHEM data into cleaned HDF5 format."""
        print("=" * 80)
        print("Preprocessing UCLCHEM Dataset")
        print("=" * 80)
        print(f"\nConfiguration:\n{OmegaConf.to_yaml(self.dataset_cfg)}")

        # Load species list
        species = np.loadtxt(
            self.dataset_cfg.species_path, dtype=str, delimiter=" ", comments=None
        ).tolist()

        # Load raw data
        input_path = self.root / self.dataset_cfg.preprocessing.input_path
        print(f"\nLoading raw data from {input_path}...")
        df = pd.read_hdf(input_path, key=self.dataset_cfg.preprocessing.input_key)
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Rename columns
        df.columns = rename_columns(df.columns)

        # Drop unwanted columns
        if self.dataset_cfg.preprocessing.columns_to_drop:
            print(
                f"\nDropping columns: {self.dataset_cfg.preprocessing.columns_to_drop}"
            )
            df = df.drop(
                columns=self.dataset_cfg.preprocessing.columns_to_drop, errors="ignore"
            )
            print(f"Remaining columns: {len(df.columns)}")

        # Set minimum radfield
        if hasattr(self.dataset_cfg.preprocessing, "min_radfield"):
            print(
                f"Setting Minimum Radfield to {self.dataset_cfg.preprocessing.min_radfield}"
            )
            df["Radfield"] = np.maximum(
                df["Radfield"], self.dataset_cfg.preprocessing.min_radfield
            )

        # Load initial abundances
        df_init = self.load_initial_abundances(species)

        # Setting the first row for each trajectory to initial abundances
        output_chunks = []
        df.sort_values(by=["Model", "Time"], inplace=True)  # type: ignore
        for _, tdf in df.groupby("Model", sort=False):
            tdf = tdf.reset_index(drop=True)

            df_init["Model"] = tdf.iloc[0]["Model"]

            tdf = pd.concat([df_init, tdf], ignore_index=True)

            physical = tdf[self.dataset_cfg.phys].shift(-1)
            physical.iloc[-1] = physical.iloc[-2]

            tdf.loc[:, self.dataset_cfg.phys] = physical.values
            output_chunks.append(tdf)

        # Combine all processed chunks and sort columns
        df = pd.concat(output_chunks, ignore_index=True)
        df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
        df.insert(0, "Index", range(len(df)))
        df = df[self.dataset_cfg.metadata + self.dataset_cfg.phys + species]

        # Apply species lower and upper clipping
        df[species] = df[species].clip(
            lower=self.dataset_cfg.abundances_clipping.lower,
            upper=self.dataset_cfg.abundances_clipping.upper,
        )

        # Split into train/validation
        print(
            f"\nSplitting data (train_split={self.dataset_cfg.preprocessing.train_split})..."
        )
        np.random.seed(self.dataset_cfg.preprocessing.seed)

        # Shuffling the tracer indices
        tracers = df["Model"].unique()
        np.random.shuffle(tracers)

        # Train/validation split based off the shuffled tracer indices
        split_idx = int(len(tracers) * self.dataset_cfg.preprocessing.train_split)

        train_tracers = tracers[:split_idx]
        val_tracers = tracers[split_idx:]

        train_df = df[df["Model"].isin(train_tracers)]
        val_df = df[df["Model"].isin(val_tracers)]

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        output_path = output_dir / "clean.h5"
        train_df.to_hdf(output_path, key="train", mode="w")
        val_df.to_hdf(output_path, key="val", mode="a")

        print("\nPreprocessing complete!")
        print(f"Saved to: {output_path}")
        print(f"  - /train: {len(train_df)} rows")
        print(f"  - /val: {len(val_df)} rows")
