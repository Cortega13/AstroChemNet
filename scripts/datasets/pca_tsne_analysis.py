"""Generate t-SNE visualizations of PCA-based data split trajectories.

This script loads the preprocessed data from the PCA-based split and creates
t-SNE visualizations using the top 50 PCA components. It generates plots for:
1. All tracers (combined training + validation)
2. Training tracers only (validation tracers removed)

Similar to the KNN analysis but focused on visualizing the data split structure.
"""

from pathlib import Path
from typing import Optional

import .data_loading as dl
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_species_list(cfg: DictConfig) -> list:
    """Returns the list of species from the configuration."""
    return np.loadtxt(
        cfg.dataset.species_path, dtype=str, delimiter=" ", comments=None
    ).tolist()


def extract_trajectories_from_hdf5(
    hdf5_path: str, key: str, cfg: DictConfig
) -> tuple[dict, list]:
    """Extract trajectories from preprocessed HDF5 data."""
    # Load the data
    df = pd.read_hdf(hdf5_path, key=key)

    # Convert to numpy for processing
    dataset_np = df.values

    # Extract trajectories (similar to original function)
    species = get_species_list(cfg)
    num_metadata = len(cfg.dataset.metadata)
    num_params = len(cfg.dataset.parameters)

    unique_models = np.unique(dataset_np[:, 1])
    combined_trajectories = {}
    final_log_density = []

    for model in unique_models:
        mask = dataset_np[:, 1] == model
        subset = dataset_np[mask]

        abundances = subset[:, -len(species) :].copy()
        log_abundances = np.log10(abundances + 1e-20)

        parameters = subset[:, num_metadata : num_metadata + num_params].copy()
        log_parameters = np.log10(np.maximum(parameters, 1e-10))

        combined_trajectories[model] = np.hstack((log_abundances, log_parameters))
        final_log_density.append(np.log10(subset[-1, num_metadata]))

    return combined_trajectories, final_log_density


def flatten_trajectories(trajectories: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten trajectories into 2D array and return model IDs."""
    data = []
    model_list = []
    for model in trajectories:
        trajectory = trajectories[model]
        flattened = trajectory.flatten()
        data.append(flattened)
        model_list.append(model)
    return np.array(data), np.array(model_list)


def plot_tsne_manifold(
    pca_data: np.ndarray,
    densities: list,
    model_ids: np.ndarray,
    val_models: Optional[set] = None,
    output_path: Optional[Path] = None,
    title_suffix: str = "",
):
    """Plot t-SNE visualization of trajectory manifold."""
    print(f"Computing t-SNE for {title_suffix}...")

    # Apply t-SNE to PCA data
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, max_iter=1000)
    tsne_data = tsne.fit_transform(pca_data)

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    if val_models is not None:
        # Separate training and validation points
        train_mask = np.array([model not in val_models for model in model_ids])
        val_mask = ~train_mask

        # Plot training points
        if np.any(train_mask):
            sc = ax.scatter(
                tsne_data[train_mask, 0],
                tsne_data[train_mask, 1],
                c=np.array(densities)[train_mask],
                s=8,
                cmap="viridis",
                alpha=0.7,
                label="Training",
                edgecolors="none",
            )

        # Plot validation points
        if np.any(val_mask):
            ax.scatter(
                tsne_data[val_mask, 0],
                tsne_data[val_mask, 1],
                c=np.array(densities)[val_mask],
                s=15,
                cmap="plasma",
                alpha=0.9,
                marker="^",
                edgecolors="black",
                linewidths=0.5,
                label="Validation",
            )

        ax.legend()
    else:
        # Plot all points with single color scheme
        sc = ax.scatter(
            tsne_data[:, 0],
            tsne_data[:, 1],
            c=densities,
            s=8,
            cmap="viridis",
            alpha=0.7,
            edgecolors="none",
        )

    # Add colorbar
    plt.colorbar(sc, ax=ax, label="log₁₀(Final Density) [cm⁻³]")

    # Labels and title
    ax.set_xlabel("t-SNE Component 1", fontsize=14)
    ax.set_ylabel("t-SNE Component 2", fontsize=14)
    ax.set_title(
        f"Trajectory t-SNE Manifold (PCA + 50 components){title_suffix}",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_path}")
    plt.close()


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Generate t-SNE visualizations of PCA-split trajectory data."""
    print("=" * 70)
    print("PCA-Split Trajectory t-SNE Analysis")
    print("=" * 70)

    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "outputs" / "temp_data" / "grav_collapse_clean.h5"
    val_indices_path = project_root / "outputs" / "temp_data" / "val_tracers.txt"
    output_dir = project_root / "outputs" / "plots" / "pca_tsne_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    # Load validation tracer IDs
    if val_indices_path.exists():
        val_models = set(np.loadtxt(val_indices_path, dtype=int))
        print(f"Loaded {len(val_models)} validation tracer IDs")
    else:
        print(
            "Warning: Validation indices file not found, proceeding without validation markers"
        )
        val_models = None

    # Load training data
    print("\nLoading training trajectories...")
    train_trajectories, train_densities = extract_trajectories_from_hdf5(
        str(data_path), "train", cfg
    )
    print(f"Loaded {len(train_trajectories)} training trajectories")

    # Load validation data
    print("Loading validation trajectories...")
    val_trajectories, val_densities = extract_trajectories_from_hdf5(
        str(data_path), "val", cfg
    )
    print(f"Loaded {len(val_trajectories)} validation trajectories")

    # Combine for full dataset analysis
    all_trajectories = {**train_trajectories, **val_trajectories}
    all_densities = train_densities + val_densities

    print(f"\nTotal trajectories: {len(all_trajectories)}")

    # Flatten all trajectories
    print("Flattening trajectories...")
    all_data, all_model_ids = flatten_trajectories(all_trajectories)

    # Apply PCA with 50 components
    print("Applying PCA (50 components)...")
    pca = PCA(n_components=50)
    pca_data = pca.fit_transform(all_data)

    # explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print(".3f")
    print(".3f")

    # Generate plots

    # 1. All tracers plot
    print("\nGenerating t-SNE plot for all tracers...")
    plot_tsne_manifold(
        pca_data,
        all_densities,
        all_model_ids,
        val_models=val_models,
        output_path=output_dir / "tsne_all_tracers.png",
        title_suffix=" - All Tracers",
    )

    # 2. Training-only plot (remove validation tracers)
    if val_models is not None:
        print("Generating t-SNE plot for training tracers only...")
        # Filter out validation data
        train_mask = np.array([model not in val_models for model in all_model_ids])
        train_pca_data = pca_data[train_mask]
        train_densities_filtered = np.array(all_densities)[train_mask]
        train_model_ids = all_model_ids[train_mask]

        plot_tsne_manifold(
            train_pca_data,
            train_densities_filtered,
            train_model_ids,
            val_models=None,  # No validation markers for training-only plot
            output_path=output_dir / "tsne_training_only.png",
            title_suffix=" - Training Only",
        )

    print(f"\n{'=' * 70}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
