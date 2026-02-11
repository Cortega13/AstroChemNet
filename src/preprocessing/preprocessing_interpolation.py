import numpy as np
import pandas as pd
import torch

# sys.path.append(".")
import AstroChemNet.utils as utils
from configs.general import GeneralConfig

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


initial_abundances = np.load("utils/initial_abundances.npy")
df_inits = pd.DataFrame(initial_abundances, columns=GeneralConfig.species)
df_inits["Radfield"] = 0
df_inits["Time"] = 0
df_inits["Av"] = 0
df_inits["gasTemp"] = 0
df_inits["Density"] = 0


df = pd.read_hdf("data/gravitational_collapse.h5", key="df", start=0)
df = df.drop(columns=["dustTemp", "dstep", "zeta", "SURFACE", "BULK"])
df.columns = utils.rename_columns(df.columns)

species = [
    col for col in df.columns if col not in GeneralConfig.metadata + GeneralConfig.phys
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


# Load train and val indices from vibecode
train_indices = np.loadtxt("scripts/preprocessing/train_indices.csv", dtype=int)
val_indices = np.loadtxt("scripts/preprocessing/val_indices.csv", dtype=int)

train_df = df[df["Model"].isin(train_indices)]
val_df = df[df["Model"].isin(val_indices)]

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df.to_hdf("data/grav_collapse_clean_int.h5", key="train", mode="w")
val_df.to_hdf("data/grav_collapse_clean_int.h5", key="val", mode="a")

print("Data split by indices and saved to data/grav_collapse_clean_int.h5")
