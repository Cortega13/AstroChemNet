"""Generate comprehensive visualizations of KNN-based trajectory predictions in PCA space."""

import json
from pathlib import Path

import ....src.data_loading as dl
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors


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
        log_parameters = np.log10(np.maximum(parameters, 1e-10))

        combined_trajectories[model] = np.hstack((log_abundances, log_parameters))
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

    # Calculate timesteps dynamically from data shape
    n_timesteps = predicted_traj.shape[1] // n_features

    # Reshape to (n_val, n_timesteps, n_features)
    pred_reshaped = predicted_traj.reshape(n_val, n_timesteps, n_features)
    actual_reshaped = actual_traj.reshape(n_val, n_timesteps, n_features)

    # Extract only species (first 333 columns) - these are in log space
    pred_species_log = np.clip(pred_reshaped[:, :, :n_species], -20, 0)
    actual_species_log = np.clip(actual_reshaped[:, :, :n_species], -20, 0)

    # Convert from log space to linear space
    pred_species = 10**pred_species_log
    actual_species = 10**actual_species_log

    # Exclude first timestep from analysis due to PCA reconstruction artifacts
    pred_species = pred_species[:, 1:, :]
    actual_species = actual_species[:, 1:, :]

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

    # Calculate timesteps dynamically from data shape
    n_timesteps = predicted_traj.shape[1] // n_features

    # Reshape to (n_val, n_timesteps, n_features)
    pred_reshaped = predicted_traj.reshape(n_val, n_timesteps, n_features)
    actual_reshaped = actual_traj.reshape(n_val, n_timesteps, n_features)

    # Extract species (first 333 columns) - in log space, clip to valid range
    pred_species_log = np.clip(pred_reshaped[:, :, :n_species], -20, 0)
    actual_species_log = np.clip(actual_reshaped[:, :, :n_species], -20, 0)

    # Convert from log space to linear space for species
    pred_species = 10**pred_species_log
    actual_species = 10**actual_species_log

    # Exclude first timestep from analysis due to PCA reconstruction artifacts
    pred_species = pred_species[:, 1:, :]
    actual_species = actual_species[:, 1:, :]

    # Calculate element-wise absolute percentage error for species only
    element_errors = (
        np.abs((actual_species - pred_species) / (actual_species + 1e-20)) * 100
    )

    # Get max error across all elements (timesteps and species) for each trajectory
    max_errors = np.max(element_errors, axis=(1, 2))

    return max_errors


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
    axes[0].set_title("Trajectory t-SNE (colored by final density)")
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


def plot_tsne_combined_density(
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    train_densities: list,
    val_densities: list,
    output_dir: Path,
    filename: str = "tsne_combined_density.png",
) -> None:
    """Plot t-SNE visualization of combined trajectories colored by final density."""
    print(f"Computing t-SNE embedding for {filename}...")

    # Combine training and validation data
    combined_pca = np.vstack([train_pca, val_pca])
    combined_densities = train_densities + val_densities

    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
    tsne_data = tsne.fit_transform(combined_pca)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Scatter plot colored by density
    sc = ax.scatter(
        tsne_data[:, 0],
        tsne_data[:, 1],
        c=combined_densities,
        s=8,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )

    # Add colorbar
    plt.colorbar(sc, ax=ax, label="log₁₀(Final Density)")

    # Labels and title
    ax.set_xlabel("t-SNE Component 1", fontsize=12)
    ax.set_ylabel("t-SNE Component 2", fontsize=12)
    ax.set_title(
        "Combined Trajectory t-SNE (colored by final density)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / filename}")


def get_species_indices(species_list: list, target_species: list) -> list:
    """Get indices of target species in species list."""
    indices = []
    missing = []
    for species in target_species:
        try:
            indices.append(species_list.index(species))
        except ValueError:
            missing.append(species)
    if missing:
        print(f"Warning: Species not found: {missing}")
    return indices


def plot_validation_trajectory_examples(
    train_pca: np.ndarray,
    val_pca: np.ndarray,
    pca: PCA,
    train_data: np.ndarray,
    val_data: np.ndarray,
    trajectory_errors: np.ndarray,
    cfg: DictConfig,
    output_dir: Path,
    n_examples: int = 3,
):
    """Plot validation trajectory examples comparing KNN predictions vs ground truth."""
    print("\nGenerating validation trajectory examples...")

    # Create output subdirectory
    examples_dir = output_dir / "trajectory_examples"
    examples_dir.mkdir(exist_ok=True)

    # Key astrochemical species to visualize
    important_species = ["H2", "H", "#C", "CO", "H2O", "C", "O", "CH4", "NH3", "N2"]
    species_list = get_species_list(cfg)
    species_indices = get_species_indices(species_list, important_species)

    # Filter to only species that were found
    valid_species = [
        important_species[i]
        for i, idx in enumerate(species_indices)
        if i < len(species_indices)
    ]

    # Get dimensions
    n_species = len(species_list)
    n_params = len(cfg.dataset.parameters)
    n_features = n_species + n_params
    n_timesteps = train_data.shape[1] // n_features

    # Parameter names (log scale)
    param_names = [
        "log₁₀(ρ) [cm⁻³]",
        "log₁₀(χ) [Habing]",
        "log₁₀(Aᵥ) [mag]",
        "log₁₀(T) [K]",
    ]

    # Select trajectories based on error distribution
    sorted_indices = np.argsort(trajectory_errors)

    example_configs = [
        ("low_error", sorted_indices[0], "Best Prediction"),
        ("medium_error", sorted_indices[len(sorted_indices) // 2], "Median Prediction"),
        (
            "high_error",
            sorted_indices[-len(sorted_indices) // 4],
            "High Error (75th percentile)",
        ),
    ]

    # Setup KNN for prediction reconstruction
    knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn.fit(train_pca)

    for filename, val_idx, title_suffix in example_configs:
        # Debug: Print actual error value for this trajectory
        print(
            f"\n{title_suffix}: Index {val_idx}, MAPE = {trajectory_errors[val_idx]:.4f}%"
        )

        # Get KNN prediction for this validation trajectory
        val_point = val_pca[val_idx : val_idx + 1]
        distances, indices = knn.kneighbors(val_point)

        # Distance-weighted average in PCA space
        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        predicted_pca = np.zeros_like(val_point)
        for i in range(len(val_point)):
            neighbor_components = train_pca[indices[i]]
            predicted_pca[i] = (weights[i][:, np.newaxis] * neighbor_components).sum(
                axis=0
            )

        # Inverse transform to get trajectories
        predicted_traj = pca.inverse_transform(predicted_pca).reshape(
            n_timesteps, n_features
        )
        actual_traj = val_data[val_idx].reshape(n_timesteps, n_features)

        # Extract species and parameters (both already in log space)
        pred_species_log = predicted_traj[:, :n_species]
        actual_species_log = actual_traj[:, :n_species]
        # Parameters are already in log space from extract_trajectories
        actual_params_log = actual_traj[:, n_species:]

        # Create dual-panel plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        timesteps = np.arange(n_timesteps)

        # Left panel: Chemical abundances
        colors = plt.cm.tab10(np.linspace(0, 1, len(species_indices)))  # type: ignore
        for _i, (species_idx, species_name, color) in enumerate(
            zip(species_indices, valid_species, colors)
        ):
            # Ground truth
            ax1.plot(
                timesteps,
                actual_species_log[:, species_idx],
                label=species_name,
                color=color,
                linewidth=2,
                linestyle="-",
                alpha=0.8,
            )
            # KNN prediction
            ax1.plot(
                timesteps,
                pred_species_log[:, species_idx],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.6,
            )

        ax1.set_xlabel("Timestep", fontsize=12)
        ax1.set_ylabel("log₁₀(Abundance)", fontsize=12)
        ax1.set_title("Chemical Evolution", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=9, loc="best", ncol=2)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(-20, 0.5)

        # Right panel: Physical parameters (ground truth only)
        colors_params = plt.cm.viridis(np.linspace(0, 1, n_params))  # type: ignore
        for i, (param_name, color) in enumerate(zip(param_names, colors_params)):
            # Ground truth only
            ax2.plot(
                timesteps,
                actual_params_log[:, i],
                label=param_name,
                color=color,
                linewidth=2,
                linestyle="-",
                alpha=0.8,
            )

        ax2.set_xlabel("Timestep", fontsize=12)
        ax2.set_ylabel("log₁₀(Parameter Value)", fontsize=12)
        ax2.set_title("Physical Conditions (log scale)", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=9, loc="best")
        ax2.grid(alpha=0.3)

        # Overall title with error information
        error_pct = trajectory_errors[val_idx]
        fig.suptitle(
            f"{title_suffix} - Trajectory {val_idx} (MAPE: {error_pct:.3f}%)",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        save_path = examples_dir / f"example_{filename}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


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

    # Exclude timestep 0 from analysis due to PCA reconstruction artifacts
    mape_per_timestep = mape_per_timestep[1:]
    p25 = p25[1:]
    p75 = p75[1:]
    p10 = p10[1:]
    p90 = p90[1:]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Linear scale
    timesteps = np.arange(1, n_timesteps)
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
    axes[0].set_title("Prediction Error (Linear Scale)")
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
    axes[1].set_title("Prediction Error (Log Scale)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_dir / "mape_vs_timestep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'mape_vs_timestep.png'}")


def load_and_prepare_data(cfg: DictConfig) -> tuple[dict, dict, list, list]:
    """Load datasets and extract trajectory data."""
    print("\nLoading datasets...")
    training_np, validation_np = dl.load_dataset(cfg.dataset)

    print("Extracting trajectories...")
    train_trajectories, train_densities = extract_trajectories(cfg, training_np)
    val_trajectories, val_densities = extract_trajectories(cfg, validation_np)

    return train_trajectories, val_trajectories, train_densities, val_densities


def compute_pca_representation(
    train_trajectories: dict, val_trajectories: dict, n_components: int = 5
) -> tuple[PCA, np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray, np.ndarray]:
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

    # Exclude first timestep from analysis due to PCA reconstruction artifacts
    pred_species_trimmed = pred_species[:, 1:, :]
    actual_species_trimmed = actual_species[:, 1:, :]

    # Per-trajectory MAPE
    trajectory_errors = (
        np.mean(
            np.abs(
                (actual_species_trimmed - pred_species_trimmed)
                / (actual_species_trimmed + 1e-20)
            ),
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
    species_errors: np.ndarray,
    species_list: list,
    output_dir: Path,
    pca: PCA,
    max_element_errors: np.ndarray,
) -> None:
    """Save species error statistics to JSON file."""
    print("Saving species error statistics...")

    sorted_indices = np.argsort(species_errors)[::-1]

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Get top 5 element errors (highest max element errors)
    sorted_element_errors = np.sort(max_element_errors)[::-1]
    top_5_element_errors = [float(err) for err in sorted_element_errors[:5]]

    statistics = {
        "pca_variance": {
            "n_components": int(pca.n_components_),
            "explained_variance_ratio": [
                float(val) for val in pca.explained_variance_ratio_
            ],
            "cumulative_explained_variance": [
                float(val) for val in cumulative_variance
            ],
            "total_variance_explained": float(cumulative_variance[-1]),
        },
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
        "top_5_element_errors": top_5_element_errors,
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
    train_data: np.ndarray,
    val_data: np.ndarray,
    cfg: DictConfig,
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

    # t-SNE with max element errors (no filtering)
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

    # MAPE evolution over timesteps
    print("Generating MAPE vs timestep plot...")
    plot_mape_vs_timestep(predicted_traj_log, actual_traj_log, output_dir)

    # Validation trajectory examples
    plot_validation_trajectory_examples(
        train_pca,
        val_pca,
        pca,
        train_data,
        val_data,
        trajectory_errors,
        cfg,
        output_dir,
    )

    # Combined t-SNE plot with density coloring
    plot_tsne_combined_density(
        train_pca,
        val_pca,
        train_densities,
        val_densities,
        output_dir,
    )


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


def run_analysis(cfg: DictConfig, output_dir: Path):
    """Run complete KNN analysis pipeline."""
    print(f"\n{'=' * 60}")
    print(f"Output Directory: {output_dir}")
    print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    train_trajectories, val_trajectories, train_densities, val_densities = (
        load_and_prepare_data(cfg)
    )

    # Compute PCA representation
    pca, train_pca, val_pca = compute_pca_representation(
        train_trajectories, val_trajectories
    )

    # Get flattened data for trajectory examples
    train_data = flatten_trajectories(train_trajectories)
    val_data = flatten_trajectories(val_trajectories)

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
    save_species_error_statistics(
        species_errors, get_species_list(cfg), output_dir, pca, max_element_errors
    )

    # Generate all visualizations
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
        train_data,
        val_data,
        cfg,
    )

    print(f"\n{'=' * 60}")
    print(f"Analysis complete. Outputs saved to: {output_dir}")
    print(f"{'=' * 60}")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entrypoint function."""
    # Set Hydra output directory to project outputs
    import os

    project_root = Path(__file__).parent.parent.parent.parent
    hydra_output_dir = project_root / "outputs" / "hydra_logs"
    os.environ["HYDRA_RUN_DIR"] = str(hydra_output_dir)
    """Generate comprehensive visualizations of KNN-based trajectory predictions."""
    print("=" * 60)
    print("KNN Trajectory Prediction Analysis")
    print("=" * 60)

    # Use absolute path relative to project root
    output_dir = (
        Path(__file__).parent.parent.parent.parent
        / "outputs"
        / "plots"
        / "knn_analysis"
    )

    run_analysis(cfg, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
