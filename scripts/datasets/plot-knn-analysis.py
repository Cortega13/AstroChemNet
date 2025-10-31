"""Generate comprehensive visualizations of KNN-based trajectory predictions in PCA space."""

import json
from pathlib import Path
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

import AstroChemNet.data_loading as dl


def get_species_list(cfg: DictConfig) -> list:
    """Returns the list of species from the configuration."""
    return np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()


def extract_trajectories(cfg: DictConfig, dataset: np.ndarray) -> tuple[dict, list]:
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

        abundances = subset[:, -len(species) :].copy()
        log_abundances = np.log10(abundances + 1e-20)

        parameters = subset[:, num_metadata : num_metadata + num_params].copy()

        combined_trajectories[model] = np.hstack((log_abundances, parameters))
        final_log_density.append(np.log10(subset[-1, num_metadata]))

    return combined_trajectories, final_log_density


def flatten_trajectories(trajectories: dict) -> np.ndarray:
    """Flatten trajectories into 2D array."""
    data = []
    for model in trajectories:
        trajectory = trajectories[model]
        flattened = trajectory.flatten()
        data.append(flattened)
    return np.array(data)


def calculate_per_species_errors(
    predicted_traj: np.ndarray,
    actual_traj: np.ndarray,
    species_list: list,
    n_timesteps: int = 298,
    n_species: int = 333,
    n_params: int = 4,
) -> np.ndarray:
    """Calculate MAPE for each species in unscaled abundance space."""
    n_val = len(predicted_traj)
    n_features = n_species + n_params

    # Reshape to (n_val, n_timesteps, n_features)
    pred_reshaped = predicted_traj.reshape(n_val, n_timesteps, n_features)
    actual_reshaped = actual_traj.reshape(n_val, n_timesteps, n_features)

    # Extract only species (first 333 columns) - these are in log space
    pred_species_log = np.clip(pred_reshaped[:, :, :n_species], -20, 0)
    actual_species_log = np.clip(actual_reshaped[:, :, :n_species], -20, 0)

    # Convert from log space to linear space
    pred_species = 10**pred_species_log
    actual_species = 10**actual_species_log

    # Calculate per-species MAPE in unscaled space
    species_errors = (
        np.mean(
            np.abs((actual_species - pred_species) / (actual_species + 1e-20)),
            axis=(0, 1),
        )
        * 100
    )

    return species_errors


def calculate_max_element_errors(
    predicted_traj: np.ndarray,
    actual_traj: np.ndarray,
    n_timesteps: int = 298,
    n_species: int = 333,
    n_params: int = 4,
) -> np.ndarray:
    """Calculate maximum element-wise error for each trajectory in unscaled space."""
    n_val = len(predicted_traj)
    n_features = n_species + n_params

    # Reshape to (n_val, n_timesteps, n_features)
    pred_reshaped = predicted_traj.reshape(n_val, n_timesteps, n_features)
    actual_reshaped = actual_traj.reshape(n_val, n_timesteps, n_features)

    # Extract species (first 333 columns) - in log space, clip to valid range
    pred_species_log = np.clip(pred_reshaped[:, :, :n_species], -20, 0)
    actual_species_log = np.clip(actual_reshaped[:, :, :n_species], -20, 0)

    # Convert from log space to linear space for species
    pred_species = 10**pred_species_log
    actual_species = 10**actual_species_log

    # Calculate element-wise absolute percentage error for species only
    element_errors = (
        np.abs((actual_species - pred_species) / (actual_species + 1e-20)) * 100
    )

    # Get max error across all elements (timesteps and species) for each trajectory
    max_errors = np.max(element_errors, axis=(1, 2))

    return max_errors


def plot_pca_variance(pca: PCA, output_dir: Path):
    """Plot explained variance ratio for PCA components."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual variance
    ax1.bar(range(1, 6), pca.explained_variance_ratio_, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_title("Variance Explained by Each PC")
    ax1.set_xticks(range(1, 6))
    ax1.grid(axis="y", alpha=0.3)

    # Cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(
        range(1, 6), cumvar, marker="o", linewidth=2, markersize=8, color="darkred"
    )
    ax2.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax2.axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, label="99% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(range(1, 6))
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_variance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'pca_variance.png'}")


def plot_tsne_with_errors(
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    errors: np.ndarray,
    train_densities: list,
    val_densities: list,
    output_dir: Path,
    error_label: str = "MAPE (%)",
    filename: str = "tsne_manifold_errors.png",
    log_scale: bool = False,
):
    """Plot t-SNE visualization with training and validation points colored by error."""
    print(f"Computing t-SNE embedding for {filename}...")
    combined_pca = np.vstack([train_pca, val_pca])
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
    tsne_data = tsne.fit_transform(combined_pca)

    train_tsne = tsne_data[: len(train_pca)]
    val_tsne = tsne_data[len(train_pca) :]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by density
    sc1 = axes[0].scatter(
        train_tsne[:, 0],
        train_tsne[:, 1],
        c=train_densities,
        s=5,
        cmap="viridis",
        alpha=0.4,
        label="Training",
    )
    axes[0].scatter(
        val_tsne[:, 0],
        val_tsne[:, 1],
        c=val_densities,
        s=10,
        cmap="viridis",
        alpha=0.8,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        label="Validation",
    )
    plt.colorbar(sc1, ax=axes[0], label="log₁₀(Final Density)")
    axes[0].set_xlabel("t-SNE Component 1")
    axes[0].set_ylabel("t-SNE Component 2")
    axes[0].set_title("Trajectory Manifold (colored by final density)")
    axes[0].legend()

    # Right: validation colored by prediction error
    axes[1].scatter(
        train_tsne[:, 0],
        train_tsne[:, 1],
        c="lightgray",
        s=3,
        alpha=0.3,
        label="Training",
    )

    # Use log scale for errors if requested
    error_values = np.log10(errors + 1e-10) if log_scale else errors

    sc2 = axes[1].scatter(
        val_tsne[:, 0],
        val_tsne[:, 1],
        c=error_values,
        s=20,
        cmap="hot_r",
        alpha=0.9,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
    )
    cbar_label = f"log₁₀({error_label})" if log_scale else error_label
    plt.colorbar(sc2, ax=axes[1], label=cbar_label)
    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    title_suffix = " (log scale)" if log_scale else ""
    axes[1].set_title(f"Validation Error{title_suffix} (darker = higher error)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def plot_knn_distance_analysis(
    train_pca: np.ndarray, val_pca: np.ndarray, errors: np.ndarray, output_dir: Path
):
    """Analyze relationship between KNN distances and prediction errors."""
    knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn.fit(train_pca)
    distances, indices = knn.kneighbors(val_pca)

    mean_distances = distances.mean(axis=1)
    min_distances = distances[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top left: Error vs mean KNN distance
    axes[0, 0].scatter(mean_distances, errors, alpha=0.5, s=10, color="steelblue")
    axes[0, 0].set_xlabel("Mean Distance to 10 Nearest Neighbors")
    axes[0, 0].set_ylabel("MAPE (%)")
    axes[0, 0].set_title("Prediction Error vs. Mean KNN Distance")
    axes[0, 0].grid(alpha=0.3)

    # Top right: Error vs nearest neighbor distance
    axes[0, 1].scatter(min_distances, errors, alpha=0.5, s=10, color="darkred")
    axes[0, 1].set_xlabel("Distance to Nearest Neighbor")
    axes[0, 1].set_ylabel("MAPE (%)")
    axes[0, 1].set_title("Prediction Error vs. Closest Neighbor Distance")
    axes[0, 1].grid(alpha=0.3)

    # Bottom left: Distribution of KNN distances
    axes[1, 0].hist(
        mean_distances, bins=50, color="green", alpha=0.7, edgecolor="black"
    )
    axes[1, 0].axvline(
        mean_distances.mean(), color="red", linestyle="--", linewidth=2, label="Mean"
    )
    axes[1, 0].axvline(
        np.median(mean_distances),
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Median",
    )
    axes[1, 0].set_xlabel("Mean Distance to Neighbors")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Distribution of Mean KNN Distances")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.3)

    # Bottom right: Error distribution
    axes[1, 1].hist(errors, bins=50, color="purple", alpha=0.7, edgecolor="black")
    axes[1, 1].axvline(
        errors.mean(), color="red", linestyle="--", linewidth=2, label="Mean"
    )
    axes[1, 1].axvline(
        np.median(errors), color="orange", linestyle="--", linewidth=2, label="Median"
    )
    axes[1, 1].set_xlabel("MAPE (%)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Distribution of Prediction Errors")
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "knn_distance_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'knn_distance_analysis.png'}")


def plot_pca_space_2d(
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    train_densities: list,
    val_densities: list,
    errors: np.ndarray,
    output_dir: Path,
):
    """Plot first two PCA components showing 1D manifold structure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: colored by density
    sc1 = axes[0].scatter(
        train_pca[:, 0],
        train_pca[:, 1],
        c=train_densities,
        s=5,
        cmap="viridis",
        alpha=0.4,
        label="Training",
    )
    axes[0].scatter(
        val_pca[:, 0],
        val_pca[:, 1],
        c=val_densities,
        s=15,
        cmap="viridis",
        alpha=0.8,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
        label="Validation",
    )
    plt.colorbar(sc1, ax=axes[0], label="log₁₀(Final Density)")
    axes[0].set_xlabel("PC1 (98.6% variance)")
    axes[0].set_ylabel("PC2 (0.8% variance)")
    axes[0].set_title("First Two Principal Components (1D Manifold Structure)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: validation colored by error
    axes[1].scatter(
        train_pca[:, 0],
        train_pca[:, 1],
        c="lightgray",
        s=3,
        alpha=0.3,
        label="Training",
    )
    sc2 = axes[1].scatter(
        val_pca[:, 0],
        val_pca[:, 1],
        c=errors,
        s=25,
        cmap="hot_r",
        alpha=0.9,
        marker="^",
        edgecolors="black",
        linewidths=0.5,
    )
    plt.colorbar(sc2, ax=axes[1], label="MAPE (%)")
    axes[1].set_xlabel("PC1 (98.6% variance)")
    axes[1].set_ylabel("PC2 (0.8% variance)")
    axes[1].set_title("Validation Error in PCA Space")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pca_space_2d.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'pca_space_2d.png'}")


def plot_neighbor_examples(
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    pca: PCA,
    train_densities: list,
    val_densities: list,
    output_dir: Path,
):
    """Show examples of validation points and their nearest neighbors."""
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(train_pca)

    # Select 4 validation examples with different characteristics
    val_indices = [
        np.argmin(val_densities),  # Lowest density
        np.argmax(val_densities),  # Highest density
        len(val_densities) // 3,  # Mid-range 1
        2 * len(val_densities) // 3,  # Mid-range 2
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, val_idx in enumerate(val_indices):
        val_point = val_pca[val_idx : val_idx + 1]
        distances, neighbor_indices = knn.kneighbors(val_point)

        # Plot in PC1-PC2 space
        axes[idx].scatter(
            train_pca[:, 0],
            train_pca[:, 1],
            c="lightgray",
            s=2,
            alpha=0.2,
            label="All training",
        )

        # Highlight neighbors
        neighbor_pca = train_pca[neighbor_indices[0]]
        neighbor_dens = [train_densities[i] for i in neighbor_indices[0]]

        for i, (npc, nd) in enumerate(zip(neighbor_pca, neighbor_dens)):
            axes[idx].scatter(
                npc[0],
                npc[1],
                s=100,
                alpha=0.7,
                label=f"Neighbor {i + 1} (ρ={nd:.2f})",
                marker="o",
                edgecolors="black",
                linewidths=1.5,
            )

        # Plot validation point
        axes[idx].scatter(
            val_point[0, 0],
            val_point[0, 1],
            s=200,
            c="red",
            marker="*",
            label=f"Val (ρ={val_densities[val_idx]:.2f})",
            edgecolors="black",
            linewidths=2,
            zorder=10,
        )

        # Draw lines to neighbors
        for npc in neighbor_pca:
            axes[idx].plot(
                [val_point[0, 0], npc[0]],
                [val_point[0, 1], npc[1]],
                "k--",
                alpha=0.3,
                linewidth=0.5,
            )

        axes[idx].set_xlabel("PC1")
        axes[idx].set_ylabel("PC2")
        axes[idx].set_title(f"Example {idx + 1}: KNN Neighbors")
        axes[idx].legend(fontsize=8, loc="best")
        axes[idx].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "neighbor_examples.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'neighbor_examples.png'}")


def plot_mape_vs_timestep(
    predicted_traj: np.ndarray,
    actual_traj: np.ndarray,
    output_dir: Path,
    n_species: int = 333,
    n_params: int = 4,
):
    """Plot MAPE evolution across timesteps."""
    n_val = len(predicted_traj)
    n_features = n_species + n_params

    # Calculate timesteps from data shape
    total_features = predicted_traj.shape[1]
    n_timesteps = total_features // n_features

    # Reshape to (n_val, n_timesteps, n_features)
    pred_reshaped = predicted_traj.reshape(n_val, n_timesteps, n_features)
    actual_reshaped = actual_traj.reshape(n_val, n_timesteps, n_features)

    # Extract species (first 333 columns) - in log space
    pred_species_log = np.clip(pred_reshaped[:, :, :n_species], -20, 0)
    actual_species_log = np.clip(actual_reshaped[:, :, :n_species], -20, 0)

    # Convert to linear space
    pred_species = 10**pred_species_log
    actual_species = 10**actual_species_log

    # Calculate MAPE per timestep (averaged over validation trajectories and species)
    mape_per_timestep = (
        np.mean(
            np.abs((actual_species - pred_species) / (actual_species + 1e-20)),
            axis=(0, 2),
        )
        * 100
    )

    # Calculate percentiles for error bands
    errors_per_traj_timestep = (
        np.mean(
            np.abs((actual_species - pred_species) / (actual_species + 1e-20)), axis=2
        )
        * 100
    )
    p25 = np.percentile(errors_per_traj_timestep, 25, axis=0)
    p75 = np.percentile(errors_per_traj_timestep, 75, axis=0)
    p10 = np.percentile(errors_per_traj_timestep, 10, axis=0)
    p90 = np.percentile(errors_per_traj_timestep, 90, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Linear scale
    timesteps = np.arange(n_timesteps)
    axes[0].plot(
        timesteps, mape_per_timestep, linewidth=2, color="darkblue", label="Mean MAPE"
    )
    axes[0].fill_between(
        timesteps, p25, p75, alpha=0.3, color="steelblue", label="25-75 percentile"
    )
    axes[0].fill_between(
        timesteps, p10, p90, alpha=0.15, color="lightblue", label="10-90 percentile"
    )
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("MAPE (%)")
    axes[0].set_title("Prediction Error Evolution (Linear Scale)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Right: Log scale
    axes[1].plot(
        timesteps, mape_per_timestep, linewidth=2, color="darkred", label="Mean MAPE"
    )
    axes[1].fill_between(
        timesteps, p25, p75, alpha=0.3, color="coral", label="25-75 percentile"
    )
    axes[1].fill_between(
        timesteps, p10, p90, alpha=0.15, color="lightsalmon", label="10-90 percentile"
    )
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_yscale("log")
    axes[1].set_title("Prediction Error Evolution (Log Scale)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "mape_vs_timestep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'mape_vs_timestep.png'}")


def load_and_prepare_data(cfg: DictConfig) -> Tuple[dict, dict, list, list]:
    """Load datasets and extract trajectory data."""
    print("\nLoading datasets...")
    training_np, validation_np = dl.load_dataset(cfg.dataset)

    print("Extracting trajectories...")
    train_trajectories, train_densities = extract_trajectories(cfg, training_np)
    val_trajectories, val_densities = extract_trajectories(cfg, validation_np)

    return train_trajectories, val_trajectories, train_densities, val_densities


def compute_pca_representation(
    train_trajectories: dict, val_trajectories: dict, n_components: int = 5
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """Apply PCA to flattened trajectory data."""
    print("Applying PCA...")
    train_data = flatten_trajectories(train_trajectories)
    val_data = flatten_trajectories(val_trajectories)

    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_data)
    val_pca = pca.transform(val_data)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    return pca, train_pca, val_pca


def compute_knn_predictions(
    train_pca: np.ndarray, val_pca: np.ndarray, n_neighbors: int = 10
) -> np.ndarray:
    """Compute KNN predictions in PCA space."""
    print("Computing KNN predictions...")
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(train_pca)
    distances, indices = knn.kneighbors(val_pca)

    weights = 1.0 / (distances + 1e-10)
    weights = weights / weights.sum(axis=1, keepdims=True)

    predicted_pca = np.zeros_like(val_pca)
    for i in range(len(val_pca)):
        neighbor_components = train_pca[indices[i]]
        predicted_pca[i] = (weights[i][:, np.newaxis] * neighbor_components).sum(axis=0)

    return predicted_pca


def calculate_error_metrics(
    cfg: DictConfig, pca: PCA, predicted_pca: np.ndarray, val_pca: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
    """Calculate comprehensive error metrics."""
    print("Calculating error metrics...")

    # Inverse transform to get trajectory data
    predicted_traj_log = pca.inverse_transform(predicted_pca)
    actual_traj_log = pca.inverse_transform(val_pca)

    # Get dimensions
    species_list = get_species_list(cfg)
    n_species = len(species_list)
    n_params = len(cfg.dataset.parameters)
    n_features_per_timestep = n_species + n_params
    n_timesteps = predicted_traj_log.shape[1] // n_features_per_timestep
    n_val = len(predicted_traj_log)

    # Reshape and calculate errors
    pred_reshaped = predicted_traj_log.reshape(
        n_val, n_timesteps, n_features_per_timestep
    )
    actual_reshaped = actual_traj_log.reshape(
        n_val, n_timesteps, n_features_per_timestep
    )

    pred_species_log = np.clip(pred_reshaped[:, :, :n_species], -20, 0)
    actual_species_log = np.clip(actual_reshaped[:, :, :n_species], -20, 0)

    pred_species = 10**pred_species_log
    actual_species = 10**actual_species_log

    # Per-trajectory MAPE
    trajectory_errors = (
        np.mean(
            np.abs((actual_species - pred_species) / (actual_species + 1e-20)),
            axis=(1, 2),
        )
        * 100
    )

    # Max element errors
    max_element_errors = calculate_max_element_errors(
        predicted_traj_log, actual_traj_log, n_timesteps, n_species, n_params
    )

    # Per-species errors
    species_errors = calculate_per_species_errors(
        predicted_traj_log,
        actual_traj_log,
        species_list,
        n_timesteps,
        n_species,
        n_params,
    )

    # Create error summary
    error_summary = {
        "trajectory_errors": {
            "mean_mape": float(trajectory_errors.mean()),
            "median_mape": float(np.median(trajectory_errors)),
            "max_mape": float(trajectory_errors.max()),
        },
        "max_element_errors": {
            "mean": float(max_element_errors.mean()),
            "median": float(np.median(max_element_errors)),
            "max": float(max_element_errors.max()),
        },
        "dimensions": {
            "n_validation_trajectories": int(n_val),
            "n_species": int(n_species),
            "n_params": int(n_params),
            "n_timesteps": int(n_timesteps),
            "total_features": int(predicted_traj_log.shape[1]),
        },
    }

    return (
        trajectory_errors,
        max_element_errors,
        species_errors,
        error_summary,
        predicted_traj_log,
        actual_traj_log,
    )


def save_species_error_statistics(
    species_errors: np.ndarray, species_list: list, output_dir: Path
) -> None:
    """Save species error statistics to JSON file."""
    print("Saving species error statistics...")

    sorted_indices = np.argsort(species_errors)[::-1]

    statistics = {
        "summary": {
            "total_species": len(species_list),
            "mean_mape": float(species_errors.mean()),
            "median_mape": float(np.median(species_errors)),
            "std_mape": float(species_errors.std()),
            "min_error": float(species_errors.min()),
            "max_error": float(species_errors.max()),
            "species_with_mape_gt_50": int(np.sum(species_errors > 50)),
            "species_with_mape_gt_100": int(np.sum(species_errors > 100)),
        },
        "worst_species": {
            "name": species_list[sorted_indices[0]],
            "mape": float(species_errors[sorted_indices[0]]),
        },
        "best_species": {
            "name": species_list[np.argmin(species_errors)],
            "mape": float(species_errors.min()),
        },
        "top_20_worst": [
            {
                "rank": i + 1,
                "species": species_list[sorted_indices[i]],
                "mape": float(species_errors[sorted_indices[i]]),
            }
            for i in range(min(20, len(species_list)))
        ],
        "all_species_errors": {
            species_list[i]: float(species_errors[i]) for i in range(len(species_list))
        },
    }

    json_path = output_dir / "species_error_statistics.json"
    with open(json_path, "w") as f:
        json.dump(statistics, f, indent=2)

    print(f"Saved: {json_path}")


def generate_visualizations(
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    pca: PCA,
    train_densities: list,
    val_densities: list,
    trajectory_errors: np.ndarray,
    max_element_errors: np.ndarray,
    predicted_traj_log: np.ndarray,
    actual_traj_log: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate t-SNE visualizations and MAPE vs timestep plot."""
    print("\nGenerating visualizations...")

    # t-SNE with regular errors
    plot_tsne_with_errors(
        train_pca,
        val_pca,
        trajectory_errors,
        train_densities,
        val_densities,
        output_dir,
        log_scale=True,
    )

    # t-SNE with filtered max element errors for better visualization
    threshold = np.percentile(max_element_errors, 99.5)
    mask = max_element_errors <= threshold
    print(
        f"Filtering max errors: removing {(~mask).sum()} outliers above {threshold:.2f}%"
    )

    plot_tsne_with_errors(
        train_pca,
        val_pca[mask],
        max_element_errors[mask],
        train_densities,
        [val_densities[i] for i in range(len(val_densities)) if mask[i]],
        output_dir,
        error_label="Max Element Error (%)",
        filename="tsne_manifold_max_errors.png",
        log_scale=True,
    )

    # MAPE evolution over timesteps
    print("Generating MAPE vs timestep plot...")
    plot_mape_vs_timestep(predicted_traj_log, actual_traj_log, output_dir)


def print_error_summary(
    error_summary: dict, species_errors: np.ndarray, species_list: list
) -> None:
    """Print comprehensive error analysis summary."""
    traj_errors = error_summary["trajectory_errors"]
    max_errors = error_summary["max_element_errors"]
    dims = error_summary["dimensions"]

    print(f"Mean MAPE (unscaled): {traj_errors['mean_mape']:.4f}%")
    print(f"Median MAPE (unscaled): {traj_errors['median_mape']:.4f}%")
    print(f"Max MAPE (unscaled): {traj_errors['max_mape']:.4f}%")

    print("\nTrajectory shape info:")
    print(f"  Validation trajectories: {dims['n_validation_trajectories']}")
    print(f"  Total flattened features: {dims['total_features']}")
    print(f"  Species: {dims['n_species']}, Params: {dims['n_params']}")
    print(f"  Features per timestep: {dims['n_species'] + dims['n_params']}")
    print(f"  Timesteps: {dims['n_timesteps']}")

    print(f"\nMean Max Element Error: {max_errors['mean']:.4f}%")
    print(f"Median Max Element Error: {max_errors['median']:.4f}%")
    print(f"Max Element Error: {max_errors['max']:.4f}%")

    # Species error summary
    sorted_indices = np.argsort(species_errors)[::-1]
    print("\n" + "=" * 60)
    print("PER-SPECIES ERROR ANALYSIS")
    print("=" * 60)
    print("\nWorst performing species:")
    print(
        f"  1. {species_list[sorted_indices[0]]}: {species_errors[sorted_indices[0]]:.2f}% MAPE"
    )

    print("\nTop 20 species with highest errors:")
    for i in range(min(20, len(species_list))):
        idx = sorted_indices[i]
        print(
            f"  {i + 1:2d}. {species_list[idx]:12s}: {species_errors[idx]:6.2f}% MAPE"
        )

    print(f"\nSpecies with MAPE > 50%: {np.sum(species_errors > 50)}")
    print(f"Species with MAPE > 100%: {np.sum(species_errors > 100)}")
    print("=" * 60)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Generate comprehensive visualizations of KNN-based trajectory predictions."""
    print("=" * 60)
    print("KNN Trajectory Prediction Analysis")
    print("=" * 60)

    # Setup
    output_dir = Path("outputs/plots/knn_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    train_trajectories, val_trajectories, train_densities, val_densities = (
        load_and_prepare_data(cfg)
    )

    # Compute PCA representation
    pca, train_pca, val_pca = compute_pca_representation(
        train_trajectories, val_trajectories
    )

    # Compute KNN predictions
    predicted_pca = compute_knn_predictions(train_pca, val_pca)

    # Calculate comprehensive error metrics
    (
        trajectory_errors,
        max_element_errors,
        species_errors,
        error_summary,
        predicted_traj_log,
        actual_traj_log,
    ) = calculate_error_metrics(cfg, pca, predicted_pca, val_pca)

    # Print analysis results
    print_error_summary(error_summary, species_errors, get_species_list(cfg))

    # Save species statistics to JSON
    save_species_error_statistics(species_errors, get_species_list(cfg), output_dir)

    # Generate visualizations
    generate_visualizations(
        train_pca,
        val_pca,
        pca,
        train_densities,
        val_densities,
        trajectory_errors,
        max_element_errors,
        predicted_traj_log,
        actual_traj_log,
        output_dir,
    )

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
