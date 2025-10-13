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

print(f"Total tracers: {len(unique_models)}")

selected_models = unique_models  # Use all tracers

tracer_species = []

for model in selected_models:
    mask = combined_np[:, 1] == model
    subset = combined_np[mask]
    species_data = subset[:, -GeneralConfig.num_species :].copy()
    log_species = np.log10(species_data + 1e-20)
    tracer_species.append(log_species.flatten())

species_data = np.array(tracer_species)

# PCA
pca = PCA(n_components=50)
pca_reduced = pca.fit_transform(species_data)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=100)
embedded = tsne.fit_transform(pca_reduced)

# Count tracers where t-SNE 1 is between -24 and -16
tsne1_values = embedded[:, 0]
count_in_strip = np.sum((tsne1_values >= -24) & (tsne1_values <= -16))

print(
    f"Number of tracers in vertical strip (t-SNE 1 between -24 and -16): {count_in_strip}"
)
