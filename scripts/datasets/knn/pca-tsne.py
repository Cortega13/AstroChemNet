"""PCA and t-SNE analysis on combined chemical abundances and physical parameter trajectories."""

from pathlib import Path
from typing import cast

import ....src.data_loading as dl
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_species_list(cfg: DictConfig) -> list:
    """Returns the list of species from the configuration."""
    return np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()


def load_datasets(cfg_data: DictConfig) -> np.ndarray:
    """Load combined training and validation datasets."""
    data_np = dl.load_dataset(cfg_data, total=True)
    data_np = cast(np.ndarray, data_np)
    return data_np


def extract_trajectories(cfg: DictConfig, dataset):
    """Extract combined trajectories (abundances + parameters) grouped by model ID."""
    species = get_species_list(cfg)
    num_metadata = len(cfg.dataset.metadata)
    num_params = len(cfg.dataset.parameters)

    unique_models = np.unique(dataset[:, 1])

    combined_trajectories = {}
    final_log_density = []

    for model in unique_models:
        mask = dataset[:, 1] == model
        subset = dataset[mask]

        # Extract log abundances
        abundances = subset[:, -len(species) :].copy()
        log_abundances = np.log10(abundances + 1e-20)

        # Extract parameters
        parameters = subset[:, num_metadata : num_metadata + num_params].copy()

        # Combine abundances and parameters
        combined_trajectories[model] = np.hstack((log_abundances, parameters))

        final_log_density.append(np.log10(subset[-1, num_metadata]))

    return combined_trajectories, final_log_density


def calculate_pca(trajectories):
    """Build from training data using PCA."""
    data = []
    model_list = []

    for model in trajectories:
        trajectory = trajectories[model]
        flattened = trajectory.flatten()
        data.append(flattened)
        model_list.append(model)

    data = np.array(data)

    if len(data) == 0:
        raise ValueError("No models found.")

    pca = PCA(n_components=5)
    reduced_data = pca.fit_transform(data)

    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print()
    print(f"Original data shape: {data.shape}")
    print(f"PCA reduced shape: {reduced_data.shape}")
    print()
    return reduced_data


def plot_tsne(reduced_data, final_log_density):
    """Plot t-SNE visualization of the reduced combined trajectories."""
    # Apply t-SNE to reduce to 2D
    print("Applying t-SNE to combined trajectories...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=100)
    tsne_data = tsne.fit_transform(reduced_data)

    # Create output directory
    output_dir = Path("outputs/plots/pca_tsne")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_data[:, 0],
        tsne_data[:, 1],
        c=final_log_density,
        s=0.2,
        cmap="viridis",
        alpha=0.8,
    )
    plt.colorbar(scatter, label="log10(Density) [H nuclei/cm³]")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(
        "t-SNE Visualization of Combined Trajectories\n(Abundances + Parameters, colored by density at last timestep)"
    )
    plt.tight_layout()

    # Save plot
    output_path = output_dir / "tsne_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved t-SNE plot to {output_path}")
    plt.close()


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function to run PCA and t-SNE on combined trajectories."""
    print("Starting PCA and t-SNE analysis on combined trajectories.")
    data_np = load_datasets(cfg.dataset)

    combined_trajectories, final_log_density = extract_trajectories(cfg, data_np)

    print("PCA on combined trajectories (abundances + parameters):")
    reduced_trajectories = calculate_pca(combined_trajectories)

    plot_tsne(reduced_trajectories, final_log_density)


if __name__ == "__main__":
    main()
