"""Generates a filtered t-SNE plot of species trajectories.

- Loads t-SNE data from 'tsne_data.csv'
- Applies filters to exclude tracers in a specific strip and those with low log density
- Creates a scatter plot colored by log density
- Saves the plot to '../plots/trajectories/pca_tsne_species_filtered.png'
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Load the saved data
data = np.loadtxt("tsne_data.csv", delimiter=",", skiprows=1)
tracer_indices = data[:, 0]
tsne1 = data[:, 1]
tsne2 = data[:, 2]
log_densities = data[:, 3]

# Identify tracers in the strip
in_strip = (tsne1 >= -24) & (tsne1 <= -16)
log_density_mask = log_densities >= 6.7
combined_mask = ~in_strip & log_density_mask

os.makedirs("../plots/trajectories", exist_ok=True)

# Plot with filters applied
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne1[combined_mask],
    tsne2[combined_mask],
    c=log_densities[combined_mask],
    cmap="viridis",
    s=1,
)
plt.colorbar(scatter, label="Log Final Density")
plt.title("PCA + t-SNE of Flattened Species Trajectories (Filtered)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/pca_tsne_species_filtered.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Filtered plot generated.")
