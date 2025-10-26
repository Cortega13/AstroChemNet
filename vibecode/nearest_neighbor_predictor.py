"""Nearest Neighbor Predictor for AstroChemNet Validation Data.

This script uses PCA-reduced training data as a vector database to predict
validation trajectories using nearest neighbor search, then calculates
mean percentage error (MAPE) between predictions and ground truth.
"""

import os
import sys
from collections import defaultdict

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.abspath(".."))
import AstroChemNet.data_loading as dl


def load_config():
    """Load Hydra configuration for emulator."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the configs directory at the project root
    config_dir = os.path.join(script_dir, "..", "configs")
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=["models=emulator"])

        # Disable struct mode to allow adding computed fields
        OmegaConf.set_struct(cfg, False)

        # Manually load computed fields (compose doesn't call __post_init__)
        # Load species from file
        species_path = cfg.dataset.species_path.replace("${working_path}", os.path.abspath(".."))
        cfg.dataset.species = np.loadtxt(
            species_path, dtype=str, delimiter=" ", comments=None
        ).tolist()
        cfg.dataset.num_species = len(cfg.dataset.species)

        # Load other computed fields
        cfg.dataset.num_metadata = len(cfg.dataset.metadata)
        cfg.dataset.num_phys = len(cfg.dataset.phys)

        # Setup columns for model
        if cfg.model.model_name == "emulator":
            cfg.model.columns = cfg.dataset.metadata + cfg.dataset.phys + cfg.dataset.species
        else:
            cfg.model.columns = cfg.dataset.species
        cfg.model.num_columns = len(cfg.model.columns)

        # Re-enable struct mode
        OmegaConf.set_struct(cfg, True)
    return cfg


def load_datasets():
    """Load combined training and validation datasets."""
    cfg = load_config()
    training_np, validation_np = dl.load_datasets(cfg.dataset, cfg.model.columns)
    data_np = np.concatenate([training_np, validation_np], axis=0)
    return data_np, cfg.dataset


def load_indices():
    """Load training and validation indices from CSV files."""
    training_indices = np.loadtxt(
        "training_indices.csv", delimiter=",", skiprows=1, dtype=int
    )
    validation_indices = np.loadtxt(
        "validation_indices.csv", delimiter=",", skiprows=1, dtype=int
    )
    return training_indices, validation_indices


def extract_trajectories_by_model(dataset, dataset_config):
    """Extract trajectories grouped by model ID."""
    unique_models = np.unique(dataset[:, 1])
    model_trajectories = {}
    model_log_trajectories = {}

    for model in unique_models:
        mask = dataset[:, 1] == model
        subset = dataset[mask]

        data = subset[:, -dataset_config.num_species :].copy()
        model_trajectories[model] = data
        model_log_trajectories[model] = np.log10(data + 1e-20)

    return model_trajectories, model_log_trajectories


def build_training_vector_database(training_log_trajectories, training_indices):
    """Build vector database from training data using PCA."""
    training_data = []
    training_sequences = {}
    model_list = []

    for model in training_indices:
        trajectory = training_log_trajectories[model]
        flattened = trajectory.flatten()
        training_data.append(flattened)
        training_sequences[model] = trajectory
        model_list.append(model)

    training_data = np.array(training_data)

    if len(training_data) == 0:
        raise ValueError("No training models found.")

    pca = PCA(n_components=50)
    training_pca = pca.fit_transform(training_data)

    print(f"Training vector database built with {len(training_indices)} models")
    print(f"Original training data shape: {training_data.shape}")
    print(f"PCA reduced shape: {training_pca.shape}")

    return training_pca, training_sequences, model_list, pca


def build_validation_vectors(validation_log_trajectories, validation_indices, pca):
    """Build validation vectors using the same PCA transform."""
    validation_data = []
    validation_sequences = {}

    for model in validation_indices:
        trajectory = validation_log_trajectories[model]
        flattened = trajectory.flatten()
        validation_data.append(flattened)
        validation_sequences[model] = trajectory

    validation_data = np.array(validation_data)

    validation_pca = pca.transform(validation_data)

    print(f"Validation vectors built with {len(validation_indices)} models")
    print(f"Original validation data shape: {validation_data.shape}")
    print(f"PCA reduced shape: {validation_pca.shape}")

    return validation_pca, validation_sequences


def predict_with_nearest_neighbors(
    training_pca,
    training_sequences,
    validation_pca,
    validation_sequences,
    model_list,
    k_neighbors=5,
):
    """Predict validation sequences using k-nearest neighbors interpolation."""
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree").fit(
        training_pca
    )

    predictions = {}
    nearest_neighbors = {}

    print(f"Finding {k_neighbors} nearest neighbors and interpolating predictions...")

    for i, (model, _) in enumerate(
        zip(validation_sequences.keys(), validation_sequences.values())
    ):
        distances, indices = nbrs.kneighbors(validation_pca[i].reshape(1, -1))

        neighbor_models = [model_list[idx] for idx in indices[0]]
        neighbor_sequences = [
            training_sequences[neighbor_model] for neighbor_model in neighbor_models
        ]
        neighbor_distances = distances[0]

        weights = 1.0 / (neighbor_distances + 1e-10)
        weights = weights / np.sum(weights)

        min_length = min(seq.shape[0] for seq in neighbor_sequences)
        interpolated_seq = np.zeros((min_length, neighbor_sequences[0].shape[1]))

        for j, seq in enumerate(neighbor_sequences):
            truncated_seq = seq[:min_length]
            interpolated_seq += weights[j] * truncated_seq

        predictions[model] = interpolated_seq
        nearest_neighbors[model] = neighbor_models

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(validation_sequences)} validation models")

    return predictions, nearest_neighbors


def calculate_mape(predictions, validation_sequences):
    """Calculate Mean Absolute Percentage Error (MAPE) for each validation model."""
    mape_scores = {}

    for model in predictions:
        predicted = predictions[model]
        actual = validation_sequences[model]

        min_timesteps = min(predicted.shape[0], actual.shape[0])
        predicted = predicted[:min_timesteps]
        actual = actual[:min_timesteps]

        epsilon = 1e-20
        actual_safe = np.where(np.abs(actual) < epsilon, epsilon, actual)

        mape = np.mean(np.abs((actual - predicted) / actual_safe)) * 100
        mape_scores[model] = mape

    return mape_scores


def print_prediction_statistics(mape_scores, nearest_neighbors):
    """Print comprehensive statistics about prediction performance."""
    mape_values = list(mape_scores.values())

    print("\n=== Nearest Neighbor Prediction Statistics ===")
    print(f"Number of predictions: {len(mape_scores)}")
    print(f"Mean MAPE: {np.mean(mape_values):.2f}%")
    print(f"Median MAPE: {np.median(mape_values):.2f}%")
    print(f"Min MAPE: {np.min(mape_values):.2f}%")
    print(f"Max MAPE: {np.max(mape_values):.2f}%")
    print(f"Std MAPE: {np.std(mape_values):.2f}%")

    print("\nMAPE Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(mape_values, p)
        print(f"  {p}th percentile: {value:.2f}%")

    all_neighbors = []
    for neighbor_list in nearest_neighbors.values():
        all_neighbors.extend(neighbor_list)
    unique_neighbors = set(all_neighbors)
    print(f"\nUnique training models used as neighbors: {len(unique_neighbors)}")

    neighbor_counts = defaultdict(int)
    for neighbor_list in nearest_neighbors.values():
        for neighbor in neighbor_list:
            neighbor_counts[neighbor] += 1

    print("\nTop 10 most frequently used training neighbors:")
    sorted_neighbors = sorted(neighbor_counts.items(), key=lambda x: x[1], reverse=True)
    for neighbor, count in sorted_neighbors[:10]:
        print(f"  Model {neighbor}: used {count} times")


def main():
    """Main function to run the nearest neighbor prediction pipeline."""
    print("Starting Nearest Neighbor Prediction Pipeline...")

    data_np, dataset_config = load_datasets()
    training_indices, validation_indices = load_indices()

    trajectories, log_trajectories = extract_trajectories_by_model(
        data_np, dataset_config
    )

    training_pca, training_sequences, model_list, pca = build_training_vector_database(
        log_trajectories, training_indices
    )

    validation_pca, validation_sequences = build_validation_vectors(
        log_trajectories, validation_indices, pca
    )

    predictions, nearest_neighbors = predict_with_nearest_neighbors(
        training_pca,
        training_sequences,
        validation_pca,
        validation_sequences,
        model_list,
        k_neighbors=10,
    )

    lin_predictions = {
        m: np.maximum(10**pred - 1e-20, 0.0) for m, pred in predictions.items()
    }
    mape_scores = calculate_mape(lin_predictions, trajectories)

    print_prediction_statistics(mape_scores, nearest_neighbors)

    print("\nNearest neighbor prediction pipeline completed!")


if __name__ == "__main__":
    main()
