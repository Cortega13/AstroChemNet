import os
import sys

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

# Save data: tracer index (model ID), tsne1, tsne2, log_density
data = np.column_stack(
    (unique_models, embedded[:, 0], embedded[:, 1], tracer_log_densities)
)
np.savetxt(
    "tsne_data.csv",
    data,
    delimiter=",",
    header="tracer_index,tsne1,tsne2,log_density",
    comments="",
)

print("Data saved to tsne_data.csv")
