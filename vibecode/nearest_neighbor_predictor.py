"""Nearest Neighbor Predictor for AstroChemNet Validation Data.

This script uses PCA-reduced training data as a vector database to predict
validation trajectories using nearest neighbor search, then calculates
mean percentage error (MAPE) between predictions and ground truth.
"""

import os
from collections import defaultdict
from typing import cast

import hydra
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

import AstroChemNet.data_loading as dl


def load_datasets(cfg_data: DictConfig) -> np.ndarray:
    """Load combined training and validation datasets."""
    data_np = dl.load_dataset(cfg_data, total=True)
    data_np = cast(np.ndarray, data_np)
    return data_np


def load_indices():
    """Load training and validation indices from CSV files."""
    training_indices = np.loadtxt(
        "vibecode/training_indices.csv", delimiter=",", skiprows=1, dtype=int
    )
    validation_indices = np.loadtxt(
        "vibecode/validation_indices.csv", delimiter=",", skiprows=1, dtype=int
    )
    return training_indices, validation_indices


def extract_trajectories_by_model(dataset, species: list):
    """Extract trajectories grouped by model ID."""
    unique_models = np.unique(dataset[:, 1])
    model_trajectories = {}
    model_log_trajectories = {}

    for model in unique_models:
        mask = dataset[:, 1] == model
        subset = dataset[mask]

        data = subset[:, -len(species) :].copy()
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

    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
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


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function to run the nearest neighbor prediction pipeline."""
    print("Starting Nearest Neighbor Prediction Pipeline...")

    data_np = load_datasets(cfg.dataset)
    training_indices, validation_indices = load_indices()
    species = np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()

    trajectories, log_trajectories = extract_trajectories_by_model(data_np, species)

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
