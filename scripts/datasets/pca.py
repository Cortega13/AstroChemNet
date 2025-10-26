"""PCA on the chemical trajectories and the physical parameter trajectories."""

from typing import cast

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA

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

    for model in unique_models:
        mask = dataset[:, 1] == model
        subset = dataset[mask]

        abundances = subset[:, -len(species) :].copy()
        abundances_trajectories[model] = np.log10(abundances + 1e-20)

        parameters = subset[:, num_metadata : num_metadata + num_params].copy()
        parameter_trajectories[model] = np.log10(parameters + 1e-4)

    return parameter_trajectories, abundances_trajectories


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

    pca = PCA(n_components=50)
    reduced_data = pca.fit_transform(data)

    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print()
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    print()
    print(f"Original data shape: {data.shape}")
    print(f"PCA reduced shape: {reduced_data.shape}")
    print()
    return reduced_data


def plot_tsne(trajectories):
    """Plot t-SNE visualization of the reduced trajectories."""


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function to run pca on abundances and parameters."""
    data_np = load_datasets(cfg.dataset)

    parameter_trajectories, abundances_trajectories = extract_trajectories(cfg, data_np)

    print("PCA on parameter trajectories:")
    reduced_param_trajectories = calculate_pca(parameter_trajectories)

    print("PCA on abundances trajectories:")
    reduced_abund_trajectories = calculate_pca(abundances_trajectories)

    plot_tsne(reduced_param_trajectories)
    plot_tsne(reduced_abund_trajectories)


if __name__ == "__main__":
    main()
