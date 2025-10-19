"""Nearest Neighbor Predictor for AstroChemNet Validation Data.

This script uses PCA-reduced training data as a vector database to predict
validation trajectories using nearest neighbor search, then calculates
mean percentage error (MAPE) between predictions and ground truth.
"""

import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def load_datasets():
    """Load training and validation datasets."""
    sys.path.append(os.path.abspath(".."))
    import AstroChemNet.data_loading as dl
    from configs.emulator import EMConfig
    from configs.general import GeneralConfig

    general_config = GeneralConfig()
    em_config = EMConfig()

    training_np, validation_np = dl.load_datasets(general_config, em_config.columns)
    return training_np, validation_np, general_config


def extract_trajectories_by_model(dataset, general_config):
    """Extract trajectories grouped by model ID."""
    unique_models = np.unique(dataset[:, 1])
    model_trajectories = {}
    model_log_densities = {}

    for model in unique_models:
        mask = dataset[:, 1] == model
        subset = dataset[mask]

        # Extract species abundances (unlogged)
        data = subset[:, -general_config.num_species :].copy()
        # Keep original abundances without log transformation
        model_trajectories[model] = data  # Shape: (timesteps, num_species)

        # Extract final log density
        final_density = subset[-1, general_config.num_metadata]
        model_log_densities[model] = np.log10(final_density)

    return model_trajectories, model_log_densities


def build_training_vector_database(training_trajectories, training_log_densities):
    """Build vector database from training data using PCA."""
    # Filter for high-density training models
    high_density_models = [
        model for model, density in training_log_densities.items() if density >= 6.7
    ]

    # Extract trajectories for high-density models
    training_data = []
    training_sequences = {}
    model_list = []

    for model in high_density_models:
        trajectory = training_trajectories[model]
        flattened = trajectory.flatten()  # Shape: (timesteps * num_species,)
        training_data.append(flattened)
        training_sequences[model] = trajectory
        model_list.append(model)

    training_data = np.array(
        training_data
    )  # Shape: (num_models, timesteps * num_species)

    # Apply PCA
    pca = PCA(n_components=150)
    training_pca = pca.fit_transform(training_data)

    print(f"Training vector database built with {len(high_density_models)} models")
    print(f"Original training data shape: {training_data.shape}")
    print(f"PCA reduced shape: {training_pca.shape}")

    return training_pca, training_sequences, model_list, pca


def build_validation_vectors(validation_trajectories, validation_log_densities, pca):
    """Build validation vectors using the same PCA transform."""
    # Filter for high-density validation models
    high_density_models = [
        model for model, density in validation_log_densities.items() if density >= 6.7
    ]

    # Extract trajectories for high-density models
    validation_data = []
    validation_sequences = {}
    model_list = []

    for model in high_density_models:
        trajectory = validation_trajectories[model]
        flattened = trajectory.flatten()
        validation_data.append(flattened)
        validation_sequences[model] = trajectory
        model_list.append(model)

    validation_data = np.array(validation_data)

    # Apply same PCA transform
    validation_pca = pca.transform(validation_data)

    print(f"Validation vectors built with {len(high_density_models)} models")
    print(f"Original validation data shape: {validation_data.shape}")
    print(f"PCA reduced shape: {validation_pca.shape}")

    return validation_pca, validation_sequences, model_list


def predict_with_nearest_neighbors(
    training_pca,
    training_sequences,
    validation_pca,
    validation_sequences,
    model_list,
    k_neighbors=5,
):
    """Predict validation sequences using k-nearest neighbors interpolation."""
    # Build nearest neighbors index
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree").fit(
        training_pca
    )

    predictions = {}
    nearest_neighbors = {}

    print(f"Finding {k_neighbors} nearest neighbors and interpolating predictions...")

    for i, (model, val_seq) in enumerate(
        zip(validation_sequences.keys(), validation_sequences.values())
    ):
        # Find k nearest neighbors
        distances, indices = nbrs.kneighbors(validation_pca[i].reshape(1, -1))

        # Get the k nearest training sequences
        neighbor_models = [model_list[idx] for idx in indices[0]]
        neighbor_sequences = [
            training_sequences[neighbor_model] for neighbor_model in neighbor_models
        ]
        neighbor_distances = distances[0]

        # Inverse distance weighting for interpolation
        weights = 1.0 / (
            neighbor_distances + 1e-10
        )  # Add small epsilon to avoid division by zero
        weights = weights / np.sum(weights)  # Normalize weights

        # Interpolate between the k nearest neighbors
        # First, ensure all sequences have the same length by taking the minimum length
        min_length = min(seq.shape[0] for seq in neighbor_sequences)
        interpolated_seq = np.zeros((min_length, neighbor_sequences[0].shape[1]))

        for j, seq in enumerate(neighbor_sequences):
            truncated_seq = seq[:min_length]  # Truncate to minimum length
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

        # Ensure same shape (in case of different timesteps)
        min_timesteps = min(predicted.shape[0], actual.shape[0])
        predicted = predicted[:min_timesteps]
        actual = actual[:min_timesteps]

        # Calculate MAPE: mean(|(actual - predicted) / actual|) * 100
        # Avoid division by zero by adding small epsilon
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
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")

    # Show distribution of MAPE scores
    print("\nMAPE Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(mape_values, p)
        print("5.2f")

    # Count unique nearest neighbors used (flatten the lists since we now have multiple neighbors per prediction)
    all_neighbors = []
    for neighbor_list in nearest_neighbors.values():
        all_neighbors.extend(neighbor_list)
    unique_neighbors = set(all_neighbors)
    print(f"\nUnique training models used as neighbors: {len(unique_neighbors)}")

    # Most frequently used neighbors
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

    # Load data
    training_np, validation_np, general_config = load_datasets()

    # Extract trajectories
    training_trajectories, training_log_densities = extract_trajectories_by_model(
        training_np, general_config
    )
    validation_trajectories, validation_log_densities = extract_trajectories_by_model(
        validation_np, general_config
    )

    # Build training vector database
    training_pca, training_sequences, model_list, pca = build_training_vector_database(
        training_trajectories, training_log_densities
    )

    # Build validation vectors
    validation_pca, validation_sequences, val_model_list = build_validation_vectors(
        validation_trajectories, validation_log_densities, pca
    )

    # Make predictions
    predictions, nearest_neighbors = predict_with_nearest_neighbors(
        training_pca,
        training_sequences,
        validation_pca,
        validation_sequences,
        model_list,
        k_neighbors=5,  # Use 5 nearest neighbors for interpolation
    )

    # Calculate errors
    mape_scores = calculate_mape(predictions, validation_sequences)

    # Print statistics
    print_prediction_statistics(mape_scores, nearest_neighbors)

    print("\nNearest neighbor prediction pipeline completed!")


if __name__ == "__main__":
    main()
