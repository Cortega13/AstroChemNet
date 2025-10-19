"""Saves training and validation indices based on t-SNE data filters.

- Loads t-SNE data from 'tsne_data.csv'
- Defines strips to exclude certain regions
- Filters tracers based on log density >= 6.7
- Assigns training indices to high density tracers not in strips
- Assigns validation indices to high density tracers in strips
- Saves indices to 'train_indices.csv' and 'val_indices.csv'
"""

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

# Train: tracers with log_density >= 6.7 and not in strips
train_mask = high_density_mask & ~in_strip
train_indices = tracer_indices[train_mask]

# Val: tracers with log_density >= 6.7 and in strips
val_mask = high_density_mask & in_strip
val_indices = tracer_indices[val_mask]

# Save indices
np.savetxt("train_indices.csv", train_indices, fmt="%d")
np.savetxt("val_indices.csv", val_indices, fmt="%d")

print("Train and validation indices saved.")

# Dummy import (commented out)
# train_indices = np.loadtxt('train_indices.csv', dtype=int)
# val_indices = np.loadtxt('val_indices.csv', dtype=int)
