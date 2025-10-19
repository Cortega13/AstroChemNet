"""Generates a t-SNE plot of physical parameters sequences.

- Loads t-SNE data from 'vibecode/phys_tsne_data.csv'
- Creates a scatter plot colored by log density
- Saves the plot to 'plots/trajectories/phys_pca_tsne.png'
"""

import os

import matplotlib.pyplot as plt
import numpy as np

# Load the saved data
data = np.loadtxt("vibecode/phys_tsne_data.csv", delimiter=",", skiprows=1)
tracer_indices = data[:, 0]
tsne1 = data[:, 1]
tsne2 = data[:, 2]
log_densities = data[:, 3]

os.makedirs("../plots/trajectories", exist_ok=True)

# Plot: t-SNE of Physical Parameters Sequences
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne1, tsne2, c=log_densities, cmap="viridis", s=1)
plt.colorbar(scatter, label="Log Final Density")
plt.title("PCA + t-SNE of Flattened Physical Parameters Sequences")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(
    "plots/trajectories/phys_pca_tsne.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Physical parameters t-SNE plot generated successfully.")
