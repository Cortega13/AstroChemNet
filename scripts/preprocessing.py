import json
import os
from typing import cast

import numpy as np
import pandas as pd
import torch

import src.utils as utils
from scripts.general import GeneralConfig

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Constants for batch processing
INTERMEDIATE_PATH = "data/uclchem_grav_intermediate.h5"
OUTPUT_PATH = "data/uclchem_grav.npy"
COLUMNS_PATH = "data/uclchem_grav_columns.json"


# Load initial abundances (small, stays in memory)
initial_abundances = np.load("outputs/preprocessed/uclchem_grav/initial_abundances.npy")
df_inits = pd.DataFrame(initial_abundances, columns=GeneralConfig.species)
df_inits["Radfield"] = 0
df_inits["Time"] = 0
df_inits["Av"] = 0
df_inits["gasTemp"] = 0
df_inits["Density"] = 0

params = ["Density", "Radfield", "Av", "gasTemp"]

# Read a small sample to determine species columns
df_sample = pd.read_hdf("data/uclchem_grav.h5", key="df", start=0, stop=100)
df_sample = df_sample.drop(columns=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"])
df_sample.columns = utils.rename_columns(list(df_sample.columns))

species = [
    col
    for col in df_sample.columns
    if col not in GeneralConfig.metadata + GeneralConfig.phys
]
species = sorted(species)
del df_sample  # Free memory

print(f"Found {len(species)} species columns")

# Load the full data - this is necessary since the HDF5 is not in table format
# But we'll process it in batches and write to intermediate file immediately
print("Loading source data...")
df = pd.read_hdf("data/uclchem_grav.h5", key="df")
df = df.drop(columns=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"])
df.columns = utils.rename_columns(list(df.columns))
df = cast(pd.DataFrame, df)
df.sort_values(by=["Model", "Time"], inplace=True)
df["Radfield"] = np.maximum(df["Radfield"], 1e-4)

print(f"Loaded {len(df)} rows")

# Get unique models
unique_models = df["Model"].unique()
num_models = len(unique_models)
print(f"Found {num_models} unique models")

# Process models in batches and write to intermediate HDF5
print("Processing models in batches...")

batch_size_models = 100  # Process 100 models at a time
num_batches = (num_models + batch_size_models - 1) // batch_size_models

with pd.HDFStore(INTERMEDIATE_PATH, mode="w") as out_store:
    for batch_idx in range(num_batches):
        start_model_idx = batch_idx * batch_size_models
        end_model_idx = min((batch_idx + 1) * batch_size_models, num_models)
        batch_models = unique_models[start_model_idx:end_model_idx]

        # Filter to this batch of models
        batch_df = df[df["Model"].isin(batch_models)]
        output_chunks = []

        for tracer, tdf in batch_df.groupby("Model", sort=False):
            tdf = tdf.reset_index(drop=True)

            df_inits["Model"] = tdf.iloc[0]["Model"]

            tdf = pd.concat([df_inits, tdf], ignore_index=True)

            physical = tdf[params].shift(-1)
            physical.iloc[-1] = physical.iloc[-2]

            tdf[params] = physical

            output_chunks.append(tdf)

        if output_chunks:
            batch_result = pd.concat(output_chunks, ignore_index=True)
            out_store.append("df", batch_result, format="table")
            del batch_result

        print(
            f"Processed batch {batch_idx + 1}/{num_batches} (models {start_model_idx + 1}-{end_model_idx})"
        )

        del batch_df, output_chunks

# Free the original dataframe
del df

print("Intermediate HDF5 file created. Converting to final format...")

# Final processing: read intermediate and convert to .npy
print("Reading intermediate file...")

with pd.HDFStore(INTERMEDIATE_PATH, mode="r") as store:
    df = store["df"]
    df = cast(pd.DataFrame, df)
    df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
    df.insert(0, "Index", range(len(df)))

    df = df[["Index", "Model", "Time"] + params + species]

    # Save data as .npy file
    print("Converting to numpy array...")
    data_array = df.to_numpy()
    np.save(OUTPUT_PATH, data_array)

    # Save column mapping as JSON
    columns_mapping = {str(i): col for i, col in enumerate(df.columns)}
    with open(COLUMNS_PATH, "w") as f:
        json.dump(columns_mapping, f, indent=2)

# Clean up intermediate file
print("Cleaning up intermediate file...")
os.remove(INTERMEDIATE_PATH)

print(f"Preprocessing complete! Output saved to {OUTPUT_PATH}")
print(f"Final array shape: {data_array.shape}")
