"""Generates three t-SNE plots of flattened species trajectories with different filters.

- Loads t-SNE data from 'tsne_data.csv'
- Defines strips to exclude certain regions
- Generates three plots: all tracers, high density tracers, and high density excluding strips
- Prints statistics on tracer counts
- Saves plots to '../plots/trajectories/' directory
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

# Define multiple strips
strips = [(-46, -44), (-31, -29), (-27, -25), (-23, -21), (-19, -17), (-15, -13)]
in_strip = np.zeros(len(tsne1), dtype=bool)
for low, high in strips:
    in_strip |= (tsne1 >= low) & (tsne1 <= high)

# Apply log density constraint
high_density_mask = log_densities >= 6.7

tracers_1 = len(tsne1)
tracers_2 = np.sum(high_density_mask)
tracers_3 = np.sum(high_density_mask & ~in_strip)
diff_2_3 = tracers_2 - tracers_3
ratio_diff_2 = diff_2_3 / tracers_2 if tracers_2 > 0 else 0

print(f"1. Total tracers: {tracers_1}")
print(f"2. Tracers with log_density >= 6.7: {tracers_2}")
print(f"3. Tracers with log_density >= 6.7 and not in strips: {tracers_3}")
print(f"Difference (2 - 3): {diff_2_3}")
print(f"Ratio of difference to 2: {ratio_diff_2:.4f}")

os.makedirs("../plots/trajectories", exist_ok=True)

# Plot 1: Everything
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne1, tsne2, c=log_densities, cmap="viridis", s=1)
plt.colorbar(scatter, label="Log Final Density")
plt.title("PCA + t-SNE of Flattened Species Trajectories (All Tracers)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/pca_tsne_all.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Plot 2: With log density >= 6.7
high_density_mask = log_densities >= 6.7
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne1[high_density_mask],
    tsne2[high_density_mask],
    c=log_densities[high_density_mask],
    cmap="viridis",
    s=1,
)
plt.colorbar(scatter, label="Log Final Density")
plt.title("PCA + t-SNE of Flattened Species Trajectories (Log Density >= 6.7)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/pca_tsne_high_density.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Plot 3: With log density >= 6.7 and not in strips
combined_mask = high_density_mask & ~in_strip
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    tsne1[combined_mask],
    tsne2[combined_mask],
    c=log_densities[combined_mask],
    cmap="viridis",
    s=1,
)
plt.colorbar(scatter, label="Log Final Density")
plt.title(
    "PCA + t-SNE of Flattened Species Trajectories (Log Density >= 6.7, Excluding Strips)"
)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/pca_tsne_high_density_excluding_strips.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Three plots generated successfully.")
