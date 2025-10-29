"""Generate comprehensive visualizations of KNN-based trajectory predictions in PCA space."""

from pathlib import Path

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
    """Calculate MAPE for each species across all timesteps and trajectories."""
    n_val = len(predicted_traj)
    n_features = n_species + n_params

    # Reshape to (n_val, n_timesteps, n_features)
    pred_reshaped = predicted_traj.reshape(n_val, n_timesteps, n_features)
    actual_reshaped = actual_traj.reshape(n_val, n_timesteps, n_features)

    # Extract only species (first 333 columns)
    pred_species = pred_reshaped[:, :, :n_species]
    actual_species = actual_reshaped[:, :, :n_species]

    # Calculate per-species MAPE averaged over all trajectories and timesteps
    species_errors = (
        np.mean(
            np.abs((actual_species - pred_species) / (actual_species + 1e-10)),
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
    """Calculate maximum element-wise error for each trajectory."""
    n_val = len(predicted_traj)
    n_features = n_species + n_params

    # Reshape to (n_val, n_timesteps, n_features)
    pred_reshaped = predicted_traj.reshape(n_val, n_timesteps, n_features)
    actual_reshaped = actual_traj.reshape(n_val, n_timesteps, n_features)

    # Calculate element-wise absolute percentage error
    element_errors = (
        np.abs((actual_reshaped - pred_reshaped) / (actual_reshaped + 1e-10)) * 100
    )

    # Get max error across all elements (timesteps and features) for each trajectory
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
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, n_iter=1000)
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


def plot_species_errors(
    species_errors: np.ndarray, species_list: list, output_dir: Path, top_n: int = 20
):
    """Plot species-level error distribution and highlight worst performers."""
    sorted_indices = np.argsort(species_errors)[::-1]
    top_indices = sorted_indices[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Histogram of all species errors
    axes[0, 0].hist(
        species_errors, bins=50, color="steelblue", alpha=0.7, edgecolor="black"
    )
    axes[0, 0].axvline(
        species_errors.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean",
    )
    axes[0, 0].axvline(
        np.median(species_errors),
        color="orange",
        linestyle="--",
        linewidth=2,
        label="Median",
    )
    axes[0, 0].set_xlabel("Species MAPE (%)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Distribution of Per-Species Prediction Errors")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Top-right: Top 20 worst species
    top_species = [species_list[i] for i in top_indices]
    top_errors = species_errors[top_indices]

    y_pos = np.arange(len(top_species))
    axes[0, 1].barh(y_pos, top_errors, color="darkred", alpha=0.7)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(top_species, fontsize=8)
    axes[0, 1].set_xlabel("MAPE (%)")
    axes[0, 1].set_title(f"Top {top_n} Species with Highest Errors")
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis="x", alpha=0.3)

    # Bottom-left: Cumulative error contribution
    sorted_errors = species_errors[sorted_indices]
    cumulative_contribution = np.cumsum(sorted_errors) / np.sum(sorted_errors) * 100
    axes[1, 0].plot(
        range(1, len(cumulative_contribution) + 1),
        cumulative_contribution,
        linewidth=2,
        color="steelblue",
    )
    axes[1, 0].axhline(
        y=50, color="red", linestyle="--", alpha=0.5, label="50% of total error"
    )
    axes[1, 0].axhline(
        y=80, color="orange", linestyle="--", alpha=0.5, label="80% of total error"
    )
    axes[1, 0].set_xlabel("Number of Species (sorted by error)")
    axes[1, 0].set_ylabel("Cumulative Error Contribution (%)")
    axes[1, 0].set_title("Cumulative Error Contribution by Species")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Bottom-right: Error statistics table
    axes[1, 1].axis("off")
    stats_text = f"""Species Error Statistics
========================

Total Species: {len(species_list)}

Mean MAPE: {species_errors.mean():.2f}%
Median MAPE: {np.median(species_errors):.2f}%
Std Dev: {species_errors.std():.2f}%

Min Error: {species_errors.min():.2f}% ({species_list[np.argmin(species_errors)]})
Max Error: {species_errors.max():.2f}% ({species_list[np.argmax(species_errors)]})

Species with MAPE > 50%: {np.sum(species_errors > 50)}
Species with MAPE > 100%: {np.sum(species_errors > 100)}

Top 3 Worst Species:
1. {species_list[top_indices[0]]}: {species_errors[top_indices[0]]:.2f}%
2. {species_list[top_indices[1]]}: {species_errors[top_indices[1]]:.2f}%
3. {species_list[top_indices[2]]}: {species_errors[top_indices[2]]:.2f}%
"""
    axes[1, 1].text(
        0.1, 0.5, stats_text, fontsize=10, verticalalignment="center", family="monospace"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "species_error_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'species_error_analysis.png'}")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Generate comprehensive visualizations of KNN-based trajectory predictions."""
    print("=" * 60)
    print("KNN Trajectory Prediction Analysis")
    print("=" * 60)

    # Create output directory
    output_dir = Path("outputs/plots/knn_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading datasets...")
    training_np, validation_np = dl.load_dataset(cfg.dataset)

    # Extract trajectories
    print("Extracting trajectories...")
    train_trajectories, train_densities = extract_trajectories(cfg, training_np)
    val_trajectories, val_densities = extract_trajectories(cfg, validation_np)

    # Flatten and apply PCA
    print("Applying PCA...")
    train_data = flatten_trajectories(train_trajectories)
    val_data = flatten_trajectories(val_trajectories)

    pca = PCA(n_components=5)
    train_pca = pca.fit_transform(train_data)
    val_pca = pca.transform(val_data)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # KNN predictions
    print("Computing KNN predictions...")
    knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn.fit(train_pca)
    distances, indices = knn.kneighbors(val_pca)

    weights = 1.0 / (distances + 1e-10)
    weights = weights / weights.sum(axis=1, keepdims=True)

    predicted_pca = np.zeros_like(val_pca)
    for i in range(len(val_pca)):
        neighbor_components = train_pca[indices[i]]
        predicted_pca[i] = (weights[i][:, np.newaxis] * neighbor_components).sum(axis=0)

    # Calculate per-trajectory errors (MAPE)
    predicted_traj = pca.inverse_transform(predicted_pca)
    actual_traj = pca.inverse_transform(val_pca)
    errors = (
        np.mean(np.abs((actual_traj - predicted_traj) / (actual_traj + 1e-10)), axis=1)
        * 100
    )

    print(f"Mean MAPE: {errors.mean():.4f}%")
    print(f"Median MAPE: {np.median(errors):.4f}%")
    print(f"Max MAPE: {errors.max():.4f}%")

    # Determine actual trajectory dimensions
    n_val = len(val_pca)
    total_features = predicted_traj.shape[1]
    species_list = get_species_list(cfg)
    n_species = len(species_list)
    n_params = len(cfg.dataset.parameters)
    n_features_per_timestep = n_species + n_params
    n_timesteps = total_features // n_features_per_timestep

    print(f"\nTrajectory shape info:")
    print(f"  Validation trajectories: {n_val}")
    print(f"  Total flattened features: {total_features}")
    print(f"  Species: {n_species}, Params: {n_params}")
    print(f"  Features per timestep: {n_features_per_timestep}")
    print(f"  Timesteps: {n_timesteps}")

    # Calculate max element errors
    print("\nCalculating max element errors...")
    max_element_errors = calculate_max_element_errors(
        predicted_traj, actual_traj, n_timesteps, n_species, n_params
    )
    print(f"Mean Max Element Error: {max_element_errors.mean():.4f}%")
    print(f"Median Max Element Error: {np.median(max_element_errors):.4f}%")
    print(f"Max Element Error: {max_element_errors.max():.4f}%")

    # Calculate per-species errors
    print("\nCalculating per-species errors...")
    species_errors = calculate_per_species_errors(
        predicted_traj, actual_traj, species_list, n_timesteps, n_species, n_params
    )

    # Print species error statistics
    sorted_indices = np.argsort(species_errors)[::-1]
    print("\n" + "=" * 60)
    print("PER-SPECIES ERROR ANALYSIS")
    print("=" * 60)
    print(f"\nWorst performing species:")
    print(
        f"  1. {species_list[sorted_indices[0]]}: {species_errors[sorted_indices[0]]:.2f}% MAPE"
    )

    print(f"\nTop 20 species with highest errors:")
    for i in range(20):
        idx = sorted_indices[i]
        print(f"  {i+1:2d}. {species_list[idx]:12s}: {species_errors[idx]:6.2f}% MAPE")

    print(f"\nSpecies with MAPE > 50%: {np.sum(species_errors > 50)}")
    print(f"Species with MAPE > 100%: {np.sum(species_errors > 100)}")
    print("=" * 60)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_pca_variance(pca, output_dir)
    plot_pca_space_2d(
        train_pca, val_pca, train_densities, val_densities, errors, output_dir
    )
    plot_knn_distance_analysis(train_pca, val_pca, errors, output_dir)
    plot_neighbor_examples(
        train_pca, val_pca, pca, train_densities, val_densities, output_dir
    )
    plot_tsne_with_errors(
        train_pca, val_pca, errors, train_densities, val_densities, output_dir
    )
    plot_tsne_with_errors(
        train_pca,
        val_pca,
        max_element_errors,
        train_densities,
        val_densities,
        output_dir,
        error_label="Max Element Error (%)",
        filename="tsne_manifold_max_errors.png",
        log_scale=True,
    )
    plot_species_errors(species_errors, species_list, output_dir)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
