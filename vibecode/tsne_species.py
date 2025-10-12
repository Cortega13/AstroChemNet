import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(".."))

import AstroChemNet.data_loading as dl
from configs.emulator import EMConfig
from configs.general import GeneralConfig

training_np, validation_np = dl.load_datasets(GeneralConfig, EMConfig.columns)

combined_np = np.concatenate([training_np, validation_np], axis=0)

del training_np, validation_np

unique_models = np.unique(combined_np[:, 1])

selected_models = unique_models  # Use all tracers

tracer_species = []
tracer_log_densities = []

for model in selected_models:
    mask = combined_np[:, 1] == model
    subset = combined_np[mask]
    species_data = subset[:, -GeneralConfig.num_species :].copy()
    log_species = np.log10(species_data + 1e-20)
    tracer_species.append(log_species.flatten())
    final_density = subset[-1, GeneralConfig.num_metadata]
    tracer_log_densities.append(np.log10(final_density))

# t-SNE on flattened species trajectories with different perplexities
perplexities = [100]
species_data = np.array(tracer_species)

os.makedirs("../plots/trajectories", exist_ok=True)

for perp in perplexities:
    tsne_species = TSNE(n_components=2, random_state=42, perplexity=perp)
    embedded_species = tsne_species.fit_transform(species_data)

    plt.figure(figsize=(10, 8))
    scatter_species = plt.scatter(
        embedded_species[:, 0],
        embedded_species[:, 1],
        c=tracer_log_densities,
        cmap="viridis",
        s=1,
    )
    plt.colorbar(scatter_species, label="Log Final Density")
    plt.title(f"t-SNE of Flattened Species Trajectories (Perplexity {perp})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(
        f"../plots/trajectories/tsne_species_trajectories_perp_{perp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
