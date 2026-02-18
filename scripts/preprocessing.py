import json
from typing import cast

import numpy as np
import pandas as pd
import torch

import src.utils as utils
from scripts.general import GeneralConfig

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


initial_abundances = np.load("outputs/preprocessed/uclchem_grav/initial_abundances.npy")
df_inits = pd.DataFrame(initial_abundances, columns=GeneralConfig.species)
df_inits["Radfield"] = 0
df_inits["Time"] = 0
df_inits["Av"] = 0
df_inits["gasTemp"] = 0
df_inits["Density"] = 0


df = pd.read_hdf("data/uclchem_grav.h5", key="df", start=0)
df = df.drop(columns=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"])
df.columns = utils.rename_columns(list(df.columns))

species = [
    col for col in df.columns if col not in GeneralConfig.metadata + GeneralConfig.phys
]
species = sorted(species)
df = cast(pd.DataFrame, df)
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
df = cast(pd.DataFrame, df)
df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
df.insert(0, "Index", range(len(df)))

df = df[["Index", "Model", "Time"] + params + species]


# Save data as .npy file
data_array = df.to_numpy()
np.save("data/uclchem_grav.npy", data_array)

# Save column mapping as JSON
columns_mapping = {str(i): col for i, col in enumerate(df.columns)}
with open("data/uclchem_grav_columns.json", "w") as f:
    json.dump(columns_mapping, f, indent=2)
