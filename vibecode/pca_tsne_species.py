import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
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
tracer_temps = []
tracer_log_radfields = []
tracer_log_avs = []

for model in selected_models:
    mask = combined_np[:, 1] == model
    subset = combined_np[mask]
    species_data = subset[:, -GeneralConfig.num_species :].copy()
    log_species = np.log10(species_data + 1e-20)
    tracer_species.append(log_species.flatten())
    final_density = subset[-1, GeneralConfig.num_metadata]
    final_radfield = subset[-1, GeneralConfig.num_metadata + 1]
    final_av = subset[-1, GeneralConfig.num_metadata + 2]
    final_temp = subset[-1, GeneralConfig.num_metadata + 3]
    tracer_log_densities.append(np.log10(final_density))
    tracer_temps.append(final_temp)
    tracer_log_avs.append(np.log10(final_av))
    tracer_log_radfields.append(np.log10(final_radfield))

species_data = np.array(tracer_species)

# PCA
pca = PCA(n_components=50)
pca_reduced = pca.fit_transform(species_data)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=100)
embedded = tsne.fit_transform(pca_reduced)

color_options = [
    (tracer_log_densities, "Log Final Density", "log_density"),
    (tracer_temps, "Final Temperature", "temp"),
    (tracer_log_radfields, "Log Final Radfield", "log_radfield"),
    (tracer_log_avs, "Log Final Av", "log_av"),
]

os.makedirs("../plots/trajectories", exist_ok=True)

for colors, label, suffix in color_options:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=colors, cmap="viridis", s=1)
    plt.colorbar(scatter, label=label)
    plt.title("PCA + t-SNE of Flattened Species Trajectories")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(
        f"../plots/trajectories/pca_tsne_species_trajectories_{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
