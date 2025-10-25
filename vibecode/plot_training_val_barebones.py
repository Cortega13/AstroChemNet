"""Barebones Training vs Validation t-SNE Plot Generator.

This script generates a t-SNE plot comparing training and validation data
for AstroChemNet species trajectories, with full statistics output.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_or_generate_tsne_data(tsne_data_path):
    """Load existing t-SNE data or generate it from scratch."""
    if not os.path.exists(tsne_data_path):
        print("t-SNE data not found, generating from scratch...")
        return generate_tsne_data(tsne_data_path)
    else:
        print("Loading existing t-SNE data...")
        return np.loadtxt(tsne_data_path, delimiter=",", skiprows=1)


def generate_tsne_data(tsne_data_path):
    """Generate t-SNE data from raw datasets."""
    # Import necessary modules for data loading
    sys.path.append(os.path.abspath(".."))
    import AstroChemNet.data_loading as dl
    from configs.emulator import EMConfig
    from configs.general import GeneralConfig

    # Initialize configs
    general_config = GeneralConfig()
    em_config = EMConfig()

    # Load datasets
    training_np, validation_np = dl.load_datasets(general_config, em_config.columns)
    combined_np = np.concatenate([training_np, validation_np], axis=0)
    del training_np, validation_np

    # Get unique models
    unique_models = np.unique(combined_np[:, 1])

    # Extract trajectories
    trajectories, log_densities = extract_trajectories(
        combined_np, unique_models, general_config
    )

    # Apply PCA and t-SNE
    embedded = apply_pca_tsne(trajectories)

    # Save t-SNE data
    data_to_save = np.column_stack(
        (unique_models, embedded[:, 0], embedded[:, 1], log_densities)
    )
    np.savetxt(
        tsne_data_path,
        data_to_save,
        delimiter=",",
        header="tracer_index,tsne1,tsne2,log_density",
        comments="",
    )
    print("t-SNE data saved.")

    return data_to_save


def extract_trajectories(combined_np, unique_models, general_config):
    """Extract and process trajectories for t-SNE."""
    trajectories = []
    log_densities = []

    for model in unique_models:
        mask = combined_np[:, 1] == model
        subset = combined_np[mask]
        data = subset[:, -general_config.num_species :].copy()
        processed = np.log10(data + 1e-20).flatten()
        trajectories.append(processed)
        final_density = subset[-1, general_config.num_metadata]
        log_densities.append(np.log10(final_density))

    return np.array(trajectories), log_densities


def apply_pca_tsne(trajectories):
    """Apply PCA followed by t-SNE dimensionality reduction."""
    print(f"Shape of data going into PCA: {trajectories.shape}")
    print(
        f"Dimension of vectors going into PCA (and thus t-SNE after PCA): {trajectories.shape[1]}"
    )

    pca = PCA(n_components=50)
    pca_reduced = pca.fit_transform(trajectories)
    print(f"Shape after PCA: {pca_reduced.shape}")
    print(
        f"Explained variance ratios for 50 components: {pca.explained_variance_ratio_}"
    )
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")

    embedded = TSNE(n_components=2, random_state=42, perplexity=100).fit_transform(
        pca_reduced
    )
    print(f"Shape after t-SNE: {embedded.shape}")

    return embedded


def apply_filters(tsne1, tsne2, log_densities):
    """Apply filtering logic for training/validation split."""
    # Define strips for validation
    strips = [
        (-46, -44),
        (-31, -29),
        (-27, -25),
        (-23, -21),
        (-19, -17),
        (-15, -13),
    ]
    in_strip = np.zeros(len(tsne1), dtype=bool)
    for low, high in strips:
        in_strip |= (tsne1 >= low) & (tsne1 <= high)

    # Apply filters
    high_density_mask = log_densities >= 6.7
    training_mask = high_density_mask & ~in_strip
    validation_mask = high_density_mask & in_strip

    return training_mask, validation_mask, high_density_mask


def print_statistics(tsne1, high_density_mask, training_mask, validation_mask):
    """Print comprehensive statistics about the data."""
    total_models = len(tsne1)
    high_density_count = np.sum(high_density_mask)
    removed_count = total_models - high_density_count
    training_count = np.sum(training_mask)
    validation_count = np.sum(validation_mask)

    print("=== t-SNE Data Statistics ===")
    print(f"Total models: {total_models}")
    print(f"High-density models (log_density >= 6.7): {high_density_count}")
    print(f"Removed models (log_density < 6.7): {removed_count}")
    print(f"Training models: {training_count}")
    print(f"Validation models: {validation_count}")
    print(
        f"Training + Validation: {training_count + validation_count} (should equal high-density count)"
    )


def generate_plot(tsne1, tsne2, training_mask, validation_mask, plots_dir):
    """Generate and save the training vs validation t-SNE plot."""
    # Initialize plot
    plt.figure(figsize=(10, 8))

    # Plot training data (blue)
    plt.scatter(
        tsne1[training_mask],
        tsne2[training_mask],
        c="blue",
        s=1,
        label="Training Data",
        alpha=0.7,
    )

    # Plot validation data (red)
    plt.scatter(
        tsne1[validation_mask],
        tsne2[validation_mask],
        c="red",
        s=1,
        label="Validation Data",
        alpha=0.7,
    )

    title = "PCA + t-SNE of Flattened Species Trajectories (Log Density >= 6.7)"
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{plots_dir}/tsne_training_vs_validation_barebones.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Barebones training vs validation t-SNE plot generated.")


def save_indices(data, training_mask, validation_mask):
    """Save training and validation indices to CSV files."""
    tracer_indices = data[:, 0].astype(int)

    training_indices = tracer_indices[training_mask]
    validation_indices = tracer_indices[validation_mask]

    np.savetxt(
        "training_indices.csv",
        training_indices,
        delimiter=",",
        header="tracer_index",
        comments="",
        fmt="%d",
    )

    np.savetxt(
        "validation_indices.csv",
        validation_indices,
        delimiter=",",
        header="tracer_index",
        comments="",
        fmt="%d",
    )

    print(
        f"Training indices saved to training_indices.csv ({len(training_indices)} tracers)"
    )
    print(
        f"Validation indices saved to validation_indices.csv ({len(validation_indices)} tracers)"
    )


def main():
    """Main function to orchestrate the t-SNE plot generation."""
    # Paths (adjust if needed)
    tsne_data_path = "tsne_data.csv"
    plots_dir = "../plots/trajectories"

    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Load or generate t-SNE data
    data = load_or_generate_tsne_data(tsne_data_path)

    # Extract coordinates
    tsne1 = data[:, 1]
    tsne2 = data[:, 2]
    log_densities = data[:, 3]

    # Apply filters
    training_mask, validation_mask, high_density_mask = apply_filters(
        tsne1, tsne2, log_densities
    )

    # Print statistics
    print_statistics(tsne1, high_density_mask, training_mask, validation_mask)

    # Save indices
    save_indices(data, training_mask, validation_mask)

    # Generate plot
    generate_plot(tsne1, tsne2, training_mask, validation_mask, plots_dir)


if __name__ == "__main__":
    main()
