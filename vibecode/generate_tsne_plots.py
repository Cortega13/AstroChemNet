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

for model in selected_models:
    mask = combined_np[:, 1] == model
    subset = combined_np[mask]
    species_data = subset[:, -GeneralConfig.num_species :].copy()
    log_species = np.log10(species_data + 1e-20)
    tracer_species.append(log_species.flatten())
    final_density = subset[-1, GeneralConfig.num_metadata]
    tracer_log_densities.append(np.log10(final_density))

species_data = np.array(tracer_species)

# PCA
pca = PCA(n_components=50)
pca_reduced = pca.fit_transform(species_data)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=100)
embedded = tsne.fit_transform(pca_reduced)

# Identify tracers in the strip
tsne1_values = embedded[:, 0]
in_strip = (tsne1_values >= -24) & (tsne1_values <= -16)
not_in_strip = ~in_strip

os.makedirs("../plots/trajectories", exist_ok=True)

# Plot all tracers
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedded[:, 0], embedded[:, 1], c=tracer_log_densities, cmap="viridis", s=1
)
plt.colorbar(scatter, label="Log Final Density")
plt.title("PCA + t-SNE of Flattened Species Trajectories (All Tracers)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/pca_tsne_species_all.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Plot without tracers in strip
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedded[not_in_strip, 0],
    embedded[not_in_strip, 1],
    c=np.array(tracer_log_densities)[not_in_strip],
    cmap="viridis",
    s=1,
)
plt.colorbar(scatter, label="Log Final Density")
plt.title("PCA + t-SNE of Flattened Species Trajectories (Excluding Strip Tracers)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/pca_tsne_species_excluding_strip.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Plots generated successfully.")
