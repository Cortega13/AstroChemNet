"""PCA to dimensionally reduce trajectories, then use K-Nearest Neighbors in PCA space to predict validation trajectories by averaging similar training trajectories."""

from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import AstroChemNet.data_loading as dl


def get_species_list(cfg: DictConfig) -> list:
    """Returns the list of species from the configuration."""
    return np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()


def load_datasets(cfg_data: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load combined training and validation datasets."""
    training_np, validation_np = dl.load_dataset(cfg_data)
    return training_np, validation_np


def extract_trajectories(cfg: DictConfig, dataset: np.ndarray) -> dict:
    """Extract combined trajectories (abundances + parameters) grouped by model ID."""
    species = get_species_list(cfg)
    num_metadata = len(cfg.dataset.metadata)
    num_params = len(cfg.dataset.parameters)

    unique_models = np.unique(dataset[:, 1])
    print(f"Number of unique models: {len(unique_models)}")

    combined_trajectories = {}

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

    # Print shape info for first trajectory
    first_model = list(combined_trajectories.keys())[0]
    print(f"Single trajectory shape: {combined_trajectories[first_model].shape}")
    print(f"  (timesteps × [species + parameters] = {combined_trajectories[first_model].shape[0]} × {combined_trajectories[first_model].shape[1]})")

    return combined_trajectories


def calculate_pca(
    trajectories: dict, pca: Optional[PCA] = None
) -> tuple[np.ndarray, PCA]:
    """Build from training data using PCA."""
    data = []
    model_list = []

    for model in trajectories:
        trajectory = trajectories[model]
        flattened = trajectory.flatten()
        data.append(flattened)
        model_list.append(model)

    data = np.array(data)
    print(f"Flattened data shape (before PCA): {data.shape}")
    print(f"  (num_trajectories × flattened_features = {data.shape[0]} × {data.shape[1]})")

    if len(data) == 0:
        raise ValueError("No models found.")

    if not pca:
        pca = PCA(n_components=5)
        reduced_data = pca.fit_transform(data)
        print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
        print(f"PCA reduced shape: {reduced_data.shape}")
        print()
    else:
        reduced_data = pca.transform(data)
        print(f"PCA transformed shape: {reduced_data.shape}")
        print()
    return reduced_data, pca


def find_knn_predictions(
    train_pca: np.ndarray, val_pca: np.ndarray, n_neighbors: int = 5
) -> np.ndarray:
    """Predict validation PCA components using K-nearest neighbors from training set."""
    print(f"Training KNN with {n_neighbors} neighbors...")
    print(f"Training PCA shape: {train_pca.shape}")
    print(f"Validation PCA shape: {val_pca.shape}")

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(train_pca)

    distances, indices = knn.kneighbors(val_pca)
    print(f"KNN distances shape: {distances.shape}")
    print(f"KNN indices shape: {indices.shape}")

    # Inverse distance weighting (avoid division by zero)
    weights = 1.0 / (distances + 1e-10)
    weights = weights / weights.sum(axis=1, keepdims=True)

    # Weighted average of neighbor PCA components
    predictions = np.zeros_like(val_pca)
    for i in range(len(val_pca)):
        neighbor_components = train_pca[indices[i]]
        predictions[i] = (weights[i][:, np.newaxis] * neighbor_components).sum(axis=0)

    print(f"Predicted PCA shape: {predictions.shape}")
    print(f"Mean KNN distance: {distances.mean():.4f}")
    print()
    return predictions


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run PCA + KNN mapping pipeline and evaluate with MAPE."""
    print("Loading datasets...")
    training_np, validation_np = load_datasets(cfg.dataset)
    print(f"Training dataset shape: {training_np.shape}")
    print(f"Validation dataset shape: {validation_np.shape}")

    print("\nExtracting training trajectories...")
    train_trajectories = extract_trajectories(cfg, training_np)

    print("\nExtracting validation trajectories...")
    val_trajectories = extract_trajectories(cfg, validation_np)

    print("\nPCA on training trajectories:")
    train_pca_components, pca = calculate_pca(train_trajectories)

    print("PCA transform validation trajectories:")
    val_pca_components, _ = calculate_pca(val_trajectories, pca=pca)

    print("Using KNN to predict validation PCA components:")
    predicted_pca_components = find_knn_predictions(
        train_pca_components, val_pca_components, n_neighbors=10
    )

    print("Inverse transforming to original trajectory space...")
    predicted_trajectories = pca.inverse_transform(predicted_pca_components)
    actual_trajectories = pca.inverse_transform(val_pca_components)
    print(f"Predicted trajectories shape: {predicted_trajectories.shape}")
    print(f"Actual trajectories shape: {actual_trajectories.shape}")

    print("\nCalculating MAPE...")
    mape = (
        np.mean(
            np.abs(
                (actual_trajectories - predicted_trajectories)
                / (actual_trajectories + 1e-10)
            )
        )
        * 100
    )
    print(f"MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()
