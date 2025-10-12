import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(".."))

import AstroChemNet.data_loading as dl
import AstroChemNet.data_processing as dp
from AstroChemNet.inference import Inference
from configs.autoencoder import AEConfig
from configs.emulator import EMConfig
from configs.general import GeneralConfig
from nn_architectures.autoencoder import Autoencoder, load_autoencoder

autoencoder = load_autoencoder(Autoencoder, GeneralConfig, AEConfig, inference=True)

processing = dp.Processing(GeneralConfig, AEConfig)

inference = Inference(
    GeneralConfig,
    processing,
    autoencoder,
)

training_np, validation_np = dl.load_datasets(GeneralConfig, EMConfig.columns)

combined_np = np.concatenate([training_np, validation_np], axis=0)

del training_np, validation_np

unique_models = np.unique(combined_np[:, 1])

selected_models = unique_models  # Use all tracers

tracer_latents = []
tracer_log_densities = []

for model in selected_models:
    mask = combined_np[:, 1] == model
    subset = combined_np[mask]
    species_data = subset[:, -GeneralConfig.num_species :].copy()
    processing.abundances_scaling(species_data)
    latents = inference.encode(species_data)
    latents_np = latents.cpu().numpy()
    tracer_latents.append(latents_np.flatten())
    final_density = subset[-1, GeneralConfig.num_metadata]
    tracer_log_densities.append(np.log10(final_density))

# t-SNE on flattened latent trajectories with different perplexities
perplexities = [100]
latent_data = np.array(tracer_latents)

os.makedirs("../plots/trajectories", exist_ok=True)

for perp in perplexities:
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    embedded = tsne.fit_transform(latent_data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedded[:, 0], embedded[:, 1], c=tracer_log_densities, cmap="viridis", s=1
    )
    plt.colorbar(scatter, label="Log Final Density")
    plt.title(f"t-SNE of Flattened Latent Trajectories (Perplexity {perp})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(
        f"../plots/trajectories/tsne_latent_trajectories_perp_{perp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
