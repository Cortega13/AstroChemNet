"""PCA on the chemical trajectories and the physical parameter trajectories."""

from pathlib import Path
from re import A
from typing import cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import AstroChemNet.data_loading as dl


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
    """Extract trajectories grouped by model ID."""
    species = get_species_list(cfg)
    num_metadata = len(cfg.dataset.metadata)
    num_params = len(cfg.dataset.parameters)

    unique_models = np.unique(dataset[:, 1])

    abundances_trajectories = {}
    parameter_trajectories = {}
    final_log_density = []

    for model in unique_models:
        mask = dataset[:, 1] == model
        subset = dataset[mask]

        abundances = subset[:, -len(species) :].copy()
        abundances_trajectories[model] = np.log10(abundances + 1e-20)

        parameters = subset[:, num_metadata : num_metadata + num_params].copy()
        parameter_trajectories[model] = parameters

        # add parameters to abundances trajectories
        abundances_trajectories[model] = np.hstack(
            (abundances_trajectories[model], parameters)
        )

        final_log_density.append(np.log10(subset[-1, num_metadata]))

    return parameter_trajectories, abundances_trajectories, final_log_density


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


def plot_tsne(reduced_data, final_log_density, name):
    """Plot t-SNE visualization of the reduced trajectories."""
    # Apply t-SNE to reduce to 2D
    print(f"Applying t-SNE to {name} data...")
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
        f"t-SNE Visualization of {name.capitalize()} Trajectories\n(colored by density at last timestep)"
    )
    plt.tight_layout()

    # Save plot
    output_path = output_dir / f"tsne_{name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved t-SNE plot to {output_path}")
    plt.close()


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function to run pca on abundances and parameters."""
    print("Starting PCA TSNE analysis.")
    data_np = load_datasets(cfg.dataset)

    parameter_trajectories, abundances_trajectories, final_log_density = (
        extract_trajectories(cfg, data_np)
    )

    print("PCA on parameter trajectories:")
    reduced_param_trajectories = calculate_pca(parameter_trajectories)

    print("PCA on abundances trajectories:")
    reduced_abund_trajectories = calculate_pca(abundances_trajectories)

    plot_tsne(reduced_param_trajectories, final_log_density, "parameters")
    plot_tsne(reduced_abund_trajectories, final_log_density, "abundances")


if __name__ == "__main__":
    main()
