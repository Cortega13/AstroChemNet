"""Generates a t-SNE plot comparing training and validation trajectories.

- Loads training and validation datasets
- Extracts trajectories for a subset of models
- Applies PCA and t-SNE to combined data
- Creates a scatter plot with training (red) and validation (blue) points
- Saves the plot to '../plots/trajectories/tsne_training_vs_validation.png'
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(".."))

import AstroChemNet.data_loading as dl
from configs.general import GeneralConfig

# Load configs
general_config = GeneralConfig()

# Load data
training_np, validation_np = dl.load_datasets(
    general_config,
    ["Index", "Model", "Time", "Density", "Radfield", "Av", "gasTemp"]
    + general_config.species,
)

# Load train and val indices
train_indices = np.loadtxt("../scripts/preprocessing/train_indices.csv", dtype=int)
val_indices = np.loadtxt("../scripts/preprocessing/val_indices.csv", dtype=int)

# Get training trajectories
train_trajectories = []
for idx in train_indices[:500]:  # Use more for better representation
    model_mask = training_np[:, 1] == idx
    model_data = training_np[model_mask]
    if len(model_data) > 0:
        model_data = model_data[np.argsort(model_data[:, 2])]  # sort by time
        abundances = model_data[:, -general_config.num_species :]
        log_abund = np.log10(abundances + 1e-20).flatten()
        train_trajectories.append(log_abund)

# Get validation trajectories
val_trajectories = []
for idx in val_indices[:200]:  # Use more for better representation
    model_mask = validation_np[:, 1] == idx
    model_data = validation_np[model_mask]
    if len(model_data) > 0:
        model_data = model_data[np.argsort(model_data[:, 2])]  # sort by time
        abundances = model_data[:, -general_config.num_species :]
        log_abund = np.log10(abundances + 1e-20).flatten()
        val_trajectories.append(log_abund)

print(f"Training trajectories: {len(train_trajectories)}")
print(f"Validation trajectories: {len(val_trajectories)}")

# Combine for joint t-SNE
combined_data = np.array(train_trajectories + val_trajectories)
pca = PCA(n_components=50)
pca_reduced = pca.fit_transform(combined_data)

n_samples = len(combined_data)
perplexity = min(30, n_samples - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
embedded = tsne.fit_transform(pca_reduced)

# Split back
n_train = len(train_trajectories)
embedded_train = embedded[:n_train]
embedded_val = embedded[n_train:]

# Plot
os.makedirs("../plots/trajectories", exist_ok=True)

plt.figure(figsize=(10, 8))

# Training data: red
plt.scatter(
    embedded_train[:, 0],
    embedded_train[:, 1],
    c="red",
    s=1,
    label="Training",
    alpha=0.7,
)

# Validation data: blue
plt.scatter(
    embedded_val[:, 0], embedded_val[:, 1], c="blue", s=1, label="Validation", alpha=0.7
)

plt.title("t-SNE: Training (Red) vs Validation (Blue)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.tight_layout()
plt.savefig(
    "../plots/trajectories/tsne_training_vs_validation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Training vs Validation t-SNE plot generated.")
