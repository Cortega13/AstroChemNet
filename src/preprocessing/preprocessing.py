"""Preprocessing script for UCLCHEM gravitational collapse dataset."""

import os
from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch

from src.configs.general import GeneralConfig

from .. import utils

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def preprocess_gravitational_collapse() -> None:
    """Preprocess gravitational collapse dataset.

    Reads raw data from data/gravitational_collapse.h5 and writes
    cleaned training and validation sets to data/grav_collapse_clean.h5.
    """
    project_root = GeneralConfig.project_root

    initial_abundances = np.load(
        os.path.join(project_root, "outputs/utils/initial_abundances.npy")
    )
    df_inits = pd.DataFrame(initial_abundances, columns=GeneralConfig.species)
    df_inits["Radfield"] = 0
    df_inits["Time"] = 0
    df_inits["Av"] = 0
    df_inits["gasTemp"] = 0
    df_inits["Density"] = 0

    df = pd.read_hdf(
        os.path.join(project_root, "data/gravitational_collapse.h5"), key="df", start=0
    )
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input dataset is not a dataframe.")

    df = df.drop(columns=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"])
    df.columns = utils.rename_columns(list(df.columns))

    species = [
        col
        for col in df.columns
        if col not in GeneralConfig.metadata + GeneralConfig.phys
    ]
    species = sorted(species)

    df.sort_values(by=["Model", "Time"], inplace=True)

    df["Radfield"] = np.maximum(df["Radfield"], 1e-4)
    output_chunks = []

    params = ["Density", "Radfield", "Av", "gasTemp"]

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

    df = df[["Index", "Model", "Time"] + params + species]

    tracers = df["Model"].unique()
    np.random.shuffle(tracers)

    # 75% train, 25% validation split
    split_idx = int(len(tracers) * 0.75)

    train_tracers = tracers[:split_idx]
    val_tracers = tracers[split_idx:]

    train_df = df[df["Model"].isin(train_tracers)]
    val_df = df[df["Model"].isin(val_tracers)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_hdf(
        os.path.join(project_root, "data/grav_collapse_clean.h5"), key="train", mode="w"
    )
    val_df.to_hdf(
        os.path.join(project_root, "data/grav_collapse_clean.h5"), key="val", mode="a"
    )


# Registry of available preprocessors
PREPROCESSORS: Dict[str, Callable[[], None]] = {
    "gravitational_collapse": preprocess_gravitational_collapse,
}


def get_preprocessor(dataset_name: str) -> Callable[[], None] | None:
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
