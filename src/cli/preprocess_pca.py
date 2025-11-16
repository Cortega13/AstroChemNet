"""CLI command for PCA-based preprocessing with interval selection for validation.

This module provides an alternative preprocessing approach that uses PCA on trajectory
data to select validation tracers from specific intervals of the principal component space.
This creates a more structured train/validation split compared to random selection.

Usage:
    astrochemnet-preprocess-pca [OPTIONS]

Examples:
    # Preprocess with PCA-based validation selection
    astrochemnet-preprocess-pca

    # Override output path
    astrochemnet-preprocess-pca dataset.preprocessing.output_path=data/pca_split.h5
"""

import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..utils import rename_columns


def load_initial_abundances(cfg: DictConfig, species: np.ndarray):
    """Loads initial abundances with uninitialized information."""
    initial_abundances = np.load(cfg.dataset.initial_abundances_path)
    df_init = pd.DataFrame(initial_abundances, columns=species)
    df_init["Radfield"] = 0
    df_init["Time"] = 0
    df_init["Av"] = 0
    df_init["gasTemp"] = 0
    df_init["Density"] = 0
    return df_init


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


def flatten_trajectories(trajectories: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten trajectories into 2D array."""
    data = []
    model_list = []
    for model in trajectories:
        trajectory = trajectories[model]
        flattened = trajectory.flatten()
        data.append(flattened)
        model_list.append(model)
    return np.array(data), np.array(model_list)


def select_validation_tracers_by_pca_intervals(
    trajectories: dict, n_intervals: int = 18, val_intervals: list = [1, 4]
) -> list:
    """Select validation tracers based on PCA intervals.

    Args:
        trajectories: Dict of trajectory data by model ID
        n_intervals: Number of intervals to divide PCA range into
        val_intervals: List of interval indices (0-based) to use for validation

    Returns:
        List of model IDs selected for validation
    """
    # Flatten trajectories
    data, model_list = flatten_trajectories(trajectories)

    # Compute PCA with 1 component
    pca = PCA(n_components=1)
    pca_values = pca.fit_transform(data).flatten()

    print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"PCA range: {pca_values.min():.4f} to {pca_values.max():.4f}")

    # Divide into intervals
    pca_min, pca_max = pca_values.min(), pca_values.max()
    interval_edges = np.linspace(pca_min, pca_max, n_intervals + 1)

    print(f"Interval edges: {interval_edges}")

    # Select tracers in specified intervals
    val_models = []
    for i, pca_val in enumerate(pca_values):
        # Find which interval this tracer belongs to
        for interval_idx in range(n_intervals):
            if (
                interval_edges[interval_idx]
                <= pca_val
                < interval_edges[interval_idx + 1]
            ):
                if interval_idx in val_intervals:
                    val_models.append(model_list[i])
                break
        # Handle the case where pca_val == pca_max (last interval)
        else:
            if n_intervals - 1 in val_intervals:
                val_models.append(model_list[i])

    print(
        f"Selected {len(val_models)} validation tracers from intervals {val_intervals}"
    )
    return val_models


def plot_pca_tsne_manifold(
    trajectories: dict,
    val_models: set,
    output_path: Path,
):
    """Generate t-SNE plot of all trajectories with PCA preprocessing."""
    print("Generating PCA + t-SNE manifold plot...")

    # Flatten trajectories
    data, model_ids = flatten_trajectories(trajectories)

    # Apply PCA with 50 components
    pca = PCA(n_components=50)
    pca_data = pca.fit_transform(data)

    print(".3f")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=100, max_iter=1000)
    tsne_data = tsne.fit_transform(pca_data)

    # Get densities
    densities = []
    for model in trajectories:
        densities.append(
            trajectories[model][-1, -4]
        )  # Last timestep, density (4th param from end)
    densities = np.log10(np.maximum(densities, 1e-10))

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Separate training and validation points
    train_mask = np.array([model not in val_models for model in model_ids])
    val_mask = ~train_mask

    # Plot training points
    if np.any(train_mask):
        sc = ax.scatter(
            tsne_data[train_mask, 0],
            tsne_data[train_mask, 1],
            c=densities[train_mask],
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
            c=densities[val_mask],
            s=15,
            cmap="plasma",
            alpha=0.9,
            marker="^",
            edgecolors="black",
            linewidths=0.5,
            label="Validation",
        )

    ax.legend()

    # Add colorbar
    plt.colorbar(sc, ax=ax, label="log₁₀(Final Density) [cm⁻³]")

    # Labels and title
    ax.set_xlabel("t-SNE Component 1", fontsize=14)
    ax.set_ylabel("t-SNE Component 2", fontsize=14)
    ax.set_title(
        "Trajectory t-SNE Manifold (PCA + 50 components)\nPCA-based Train/Validation Split",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved manifold plot: {output_path}")


def preprocess_dataset_pca(cfg: DictConfig):
    """Preprocess raw UCLCHEM data with PCA-based validation selection.

    Args:
        cfg: Hydra configuration with dataset parameters.
    """
    print("=" * 80)
    print("Preprocessing UCLCHEM Dataset (PCA-based Validation Selection)")
    print("=" * 80)
    print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

    # Load species list
    species_list = get_species_list(cfg)
    species_array = np.array(species_list)

    # Load raw data
    print(f"\nLoading raw data from {cfg.dataset.preprocessing.input_path}...")
    df = pd.read_hdf(
        cfg.dataset.preprocessing.input_path, key=cfg.dataset.preprocessing.input_key
    )
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Rename columns
    df.columns = rename_columns(df.columns)

    # Drop unwanted columns
    if cfg.dataset.preprocessing.columns_to_drop:
        print(f"\nDropping columns: {cfg.dataset.preprocessing.columns_to_drop}")
        df = df.drop(columns=cfg.dataset.preprocessing.columns_to_drop, errors="ignore")
        print(f"Remaining columns: {len(df.columns)}")

    # Set minimum radfield
    if hasattr(cfg.dataset.preprocessing, "min_radfield"):
        print(f"Setting Minimum Radfield to {cfg.dataset.preprocessing.min_radfield}")
        df["Radfield"] = np.maximum(
            df["Radfield"], cfg.dataset.preprocessing.min_radfield
        )

    # Load initial abundances
    df_init = load_initial_abundances(cfg, species_array)

    # Setting the first row for each trajectory to initial abundances
    output_chunks = []
    df.sort_values(by=["Model", "Time"], inplace=True)  # type: ignore
    for _, tdf in df.groupby("Model", sort=False):
        tdf = tdf.reset_index(drop=True)

        df_init["Model"] = tdf.iloc[0]["Model"]

        tdf = pd.concat([df_init, tdf], ignore_index=True)

        physical = tdf[cfg.dataset.parameters].shift(-1)
        physical.iloc[-1] = physical.iloc[-2]

        tdf.loc[:, cfg.dataset.parameters] = physical.values
        output_chunks.append(tdf)

    # Combine all processed chunks and sort columns
    df = pd.concat(output_chunks, ignore_index=True)
    df = df.sort_values(by=["Model", "Time"]).reset_index(drop=True)
    df.insert(0, "Index", range(len(df)))
    df = df[cfg.dataset.metadata + cfg.dataset.parameters + species_list]

    # Apply species lower and upper clipping
    df[species_list] = df[species_list].clip(
        lower=cfg.dataset.abundances_clipping.lower,
        upper=cfg.dataset.abundances_clipping.upper,
    )

    # Convert to numpy for trajectory extraction
    dataset_np = df.values

    # Extract trajectories for PCA analysis
    print("\nExtracting trajectories for PCA analysis...")
    trajectories, _ = extract_trajectories(cfg, dataset_np)

    # Select validation tracers using PCA intervals
    print("\nSelecting validation tracers based on PCA intervals...")
    val_tracers = select_validation_tracers_by_pca_intervals(
        trajectories, n_intervals=18, val_intervals=[2, 4, 6, 9, 11, 13]
    )

    # Split into train/validation
    print("\nSplitting data based on PCA-selected validation tracers...")
    train_tracers = sorted([m for m in df["Model"].unique() if m not in val_tracers])
    val_tracers = sorted(val_tracers)

    train_df = df[df["Model"].isin(train_tracers)]
    val_df = df[df["Model"].isin(val_tracers)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_hdf(cfg.dataset.preprocessing.output_path, key="train", mode="w")
    val_df.to_hdf(cfg.dataset.preprocessing.output_path, key="val", mode="a")

    # Save tracer indices to txt files
    output_dir = Path(cfg.dataset.preprocessing.output_path).parent
    np.savetxt(output_dir / "train_tracers.txt", train_tracers, fmt="%d")
    np.savetxt(output_dir / "val_tracers.txt", val_tracers, fmt="%d")
    print(
        f"Saved tracer indices to {output_dir}/train_tracers.txt and {output_dir}/val_tracers.txt"
    )

    # Generate t-SNE manifold plot
    plot_pca_tsne_manifold(
        trajectories, set(val_tracers), output_dir / "pca_tsne_manifold.png"
    )

    print("\nPreprocessing complete!")
    print(f"Saved to: {cfg.dataset.preprocessing.output_path}")
    print(f"  - /train: {len(train_df)} rows ({len(train_tracers)} tracers)")
    print(f"  - /val: {len(val_df)} rows ({len(val_tracers)} tracers)")
    print(f"  - Manifold plot: {output_dir}/pca_tsne_manifold.png")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for PCA-based data preprocessing CLI command.

    Args:
        cfg: Hydra configuration object from config.yaml with CLI overrides.
    """
    # Run preprocessing
    preprocess_dataset_pca(cfg)


if __name__ == "__main__":
    main()
