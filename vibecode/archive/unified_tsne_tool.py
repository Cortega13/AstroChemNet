"""Unified t-SNE Tool for AstroChemNet Data Analysis.

This script provides a comprehensive tool for processing and visualizing t-SNE data
for species trajectories and physical parameters in AstroChemNet.

Usage Examples:
    # Save t-SNE data for species trajectories
    python unified_tsne_tool.py save-species-tsne

    # Save t-SNE data for physical parameters
    python unified_tsne_tool.py save-phys-tsne

    # Save training/validation indices based on filters
    python unified_tsne_tool.py save-indices

    # Generate filtered species trajectory plot
    python unified_tsne_tool.py plot-filtered

    # Generate physical parameters t-SNE plot
    python unified_tsne_tool.py plot-phys

    # Generate three comparison plots (all, high density, high density excluding strips)
    python unified_tsne_tool.py plot-three

    # Generate training vs validation t-SNE plot
    python unified_tsne_tool.py plot-training-val

Available Modes:
- save-species-tsne: Save t-SNE data for species trajectories to tsne_data.csv
- save-phys-tsne: Save t-SNE data for physical parameters to vibecode/phys_tsne_data.csv
- save-indices: Save training/validation indices based on t-SNE filters
- plot-filtered: Generate filtered species trajectory plot with density-based filtering
- plot-phys: Generate physical parameters t-SNE plot colored by log density
- plot-three: Generate three comparison plots showing different filtering levels
- plot-training-val: Generate t-SNE plot comparing training and validation trajectories
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(".."))

import AstroChemNet.data_loading as dl
from configs.autoencoder import AEConfig
from configs.emulator import EMConfig
from configs.general import GeneralConfig
from nn_architectures.autoencoder import Autoencoder, load_autoencoder
from nn_architectures.emulator import Emulator
from src.AstroChemNet.data_processing import Processing
from src.AstroChemNet.inference import Inference


class TSNETool:
    """Unified tool for t-SNE operations on AstroChemNet data."""

    def __init__(self):
        """Initialize configurations and paths."""
        self.general_config = GeneralConfig()
        self.ae_config = AEConfig()
        self.em_config = EMConfig()

        # Common paths
        self.tsne_data_path = "tsne_data.csv"
        self.phys_tsne_data_path = "vibecode/phys_tsne_data.csv"
        self.train_indices_path = "train_indices.csv"
        self.val_indices_path = "val_indices.csv"
        self.plots_dir = "../plots/trajectories"

        # Ensure plots directory exists
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_datasets(self):
        """Load training and validation datasets."""
        return dl.load_datasets(self.general_config, self.em_config.columns)

    def load_tsne_data(self, path):
        """Load t-SNE data from CSV file."""
        return np.loadtxt(path, delimiter=",", skiprows=1)

    def save_tsne_data(self, data, path, header):
        """Save t-SNE data to CSV file."""
        np.savetxt(path, data, delimiter=",", header=header, comments="")

    def apply_pca_tsne(self, data, n_components_pca=50, n_components_tsne=2):
        """Apply PCA followed by t-SNE dimensionality reduction.

        - Reduces data dimensionality using PCA with 50 components
        - Applies t-SNE with fixed perplexity of 100 for 2D embedding
        - Returns embedded coordinates for visualization
        """
        print(f"Shape of data going into PCA: {data.shape}")
        print(
            f"Dimension of vectors going into PCA (and thus t-SNE after PCA): {data.shape[1]}"
        )
        pca = PCA(n_components=n_components_pca)
        pca_reduced = pca.fit_transform(data)
        print(f"Shape after PCA: {pca_reduced.shape}")
        print(
            f"Explained variance ratios for {n_components_pca} components: {pca.explained_variance_ratio_}"
        )
        print(
            f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}"
        )
        embedded = TSNE(
            n_components=n_components_tsne, random_state=42, perplexity=100
        ).fit_transform(pca_reduced)
        print(f"Shape after t-SNE: {embedded.shape}")
        return embedded

    def get_unique_models(self, combined_np):
        """Get unique model IDs from combined dataset.

        - Extracts model IDs from the second column of the dataset
        - Returns sorted array of unique model identifiers
        """
        return np.unique(combined_np[:, 1])

    def extract_trajectories(self, combined_np, selected_models, data_type="species"):
        """Extract and process trajectories for t-SNE.

        - Iterates through selected model IDs
        - Extracts species abundances or physical parameters based on data_type
        - Applies log transformation for species data
        - Flattens trajectories and computes log final densities
        - Returns processed trajectories and density values
        """
        trajectories = []
        log_densities = []

        for model in selected_models:
            mask = combined_np[:, 1] == model
            subset = combined_np[mask]

            if data_type == "species":
                data = subset[:, -self.general_config.num_species :].copy()
                processed = np.log10(data + 1e-20).flatten()
            else:  # phys
                data = subset[
                    :,
                    self.general_config.num_metadata : self.general_config.num_metadata
                    + self.general_config.num_phys,
                ].copy()
                processed = data.flatten()
            trajectories.append(processed)
            final_density = subset[-1, self.general_config.num_metadata]
            log_densities.append(np.log10(final_density))

        return np.array(trajectories), log_densities

    def save_tsne_data_generic(self, data_type="species"):
        """Save t-SNE data for species or physical parameters.

        - Loads and combines training/validation datasets
        - Extracts trajectories for all unique models
        - Applies PCA and t-SNE dimensionality reduction
        - Saves results to CSV with model IDs, coordinates, and densities
        """
        print(f"Loading datasets for {data_type} t-SNE...")
        training_np, validation_np = self.load_datasets()
        combined_np = np.concatenate([training_np, validation_np], axis=0)
        del training_np, validation_np

        unique_models = self.get_unique_models(combined_np)
        trajectories, log_densities = self.extract_trajectories(
            combined_np, unique_models, data_type
        )

        print("Applying PCA and t-SNE...")
        embedded = self.apply_pca_tsne(trajectories)

        path = (
            self.tsne_data_path if data_type == "species" else self.phys_tsne_data_path
        )
        data = np.column_stack(
            (unique_models, embedded[:, 0], embedded[:, 1], log_densities)
        )
        self.save_tsne_data(data, path, "tracer_index,tsne1,tsne2,log_density")
        print(f"{data_type.title()} t-SNE data saved to {path}")

    def save_species_tsne(self):
        """Save t-SNE data for species trajectories."""
        self.save_tsne_data_generic("species")

    def save_phys_tsne(self):
        """Save t-SNE data for physical parameters."""
        self.save_tsne_data_generic("phys")

    def save_indices(self):
        """Save training and validation indices based on filters.

        - Loads existing t-SNE data for species trajectories
        - Defines strip regions to exclude from training
        - Filters models by log density >= 6.7
        - Assigns high-density models outside strips to training
        - Assigns high-density models inside strips to validation
        - Saves filtered indices to CSV files
        """
        print("Loading t-SNE data for index generation...")
        data = self.load_tsne_data(self.tsne_data_path)
        tracer_indices = data[:, 0]
        tsne1 = data[:, 1]
        log_densities = data[:, 3]

        # Define strips
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
        train_mask = high_density_mask & ~in_strip
        val_mask = high_density_mask & in_strip

        train_indices = tracer_indices[train_mask]
        val_indices = tracer_indices[val_mask]

        np.savetxt(self.train_indices_path, train_indices, fmt="%d")
        np.savetxt(self.val_indices_path, val_indices, fmt="%d")
        print("Training and validation indices saved.")

    def plot_tsne_generic(self, path, title, filename):
        """Generate generic t-SNE plot.

        - Loads t-SNE data from specified path
        - Creates scatter plot colored by log density
        - Saves high-resolution plot to plots directory
        """
        print(f"Generating {filename} plot...")
        data = self.load_tsne_data(path)
        tsne1, tsne2, log_densities = data[:, 1], data[:, 2], data[:, 3]

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne1, tsne2, c=log_densities, cmap="viridis", s=1)
        plt.colorbar(scatter, label="Log Final Density")
        plt.title(title)
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/{filename}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"{filename} plot generated.")

    def plot_filtered(self):
        """Generate filtered species trajectory plot.

        - Loads species t-SNE data
        - Applies filters to exclude strip regions and low-density models
        - Creates scatter plot of filtered trajectories colored by density
        - Saves high-resolution plot to plots directory
        """
        data = self.load_tsne_data(self.tsne_data_path)
        tsne1, tsne2, log_densities = data[:, 1], data[:, 2], data[:, 3]
        in_strip = (tsne1 >= -24) & (tsne1 <= -16)
        combined_mask = ~in_strip & (log_densities >= 6.7)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne1[combined_mask],
            tsne2[combined_mask],
            c=log_densities[combined_mask],
            cmap="viridis",
            s=1,
        )
        plt.colorbar(scatter, label="Log Final Density")
        plt.title("PCA + t-SNE of Flattened Species Trajectories (Filtered)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(
            f"{self.plots_dir}/pca_tsne_species_filtered.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("Filtered plot generated.")

    def plot_phys(self):
        """Generate physical parameters t-SNE plot."""
        self.plot_tsne_generic(
            self.phys_tsne_data_path,
            "PCA + t-SNE of Flattened Physical Parameters Sequences",
            "phys_pca_tsne",
        )

    def plot_three(self):
        """Generate three comparison plots.

        - Loads species t-SNE data and defines strip regions
        - Computes statistics on tracer counts with different filters
        - Creates three plots: all data, high density, and high density excluding strips
        - Saves all plots with high resolution to plots directory
        """
        print("Generating three comparison plots...")
        data = self.load_tsne_data(self.tsne_data_path)
        tsne1, tsne2, log_densities = data[:, 1], data[:, 2], data[:, 3]

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

        high_density_mask = log_densities >= 6.7
        tracers_1, tracers_2, tracers_3 = (
            len(tsne1),
            int(np.sum(high_density_mask)),
            int(np.sum(high_density_mask & ~in_strip)),
        )
        diff_2_3 = tracers_2 - tracers_3
        ratio_diff_2 = diff_2_3 / tracers_2 if tracers_2 > 0 else 0

        print(
            f"1. Total tracers: {tracers_1}\n2. Tracers with log_density >= 6.7: {tracers_2}\n3. Tracers with log_density >= 6.7 and not in strips: {tracers_3}\nDifference (2 - 3): {diff_2_3}\nRatio of difference to 2: {ratio_diff_2:.4f}"
        )

        # All data plot (train t-SNE fresh on all training + validation data, no constraints)
        print("Computing t-SNE fresh on all training and validation data...")
        training_np, validation_np = self.load_datasets()
        combined_np = np.concatenate([training_np, validation_np], axis=0)
        del training_np, validation_np

        unique_models = self.get_unique_models(combined_np)
        all_trajectories, all_log_densities = self.extract_trajectories(
            combined_np, unique_models, "species"
        )

        embedded_all = self.apply_pca_tsne(all_trajectories)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embedded_all[:, 0],
            embedded_all[:, 1],
            c=all_log_densities,
            cmap="viridis",
            s=1,
        )
        plt.colorbar(scatter, label="Log Final Density")
        plt.title(
            "PCA + t-SNE of Flattened Species Trajectories (All Data - Fresh Training)"
        )
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(f"{self.plots_dir}/pca_tsne_all.png", dpi=300, bbox_inches="tight")
        plt.close()

        # High density plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            tsne1[high_density_mask],
            tsne2[high_density_mask],
            c=log_densities[high_density_mask],
            cmap="viridis",
            s=1,
        )
        plt.colorbar(scatter, label="Log Final Density")
        plt.title("PCA + t-SNE of Flattened Species Trajectories (Log Density >= 6.7)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.savefig(
            f"{self.plots_dir}/pca_tsne_high_density.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # High density with strips: blue for training (non-strips), red for validation (strips)
        plt.figure(figsize=(10, 8))
        # Training data (non-strips): blue
        training_mask = high_density_mask & ~in_strip
        plt.scatter(
            tsne1[training_mask],
            tsne2[training_mask],
            c="blue",
            s=1,
            label="Training Data",
            alpha=0.7,
        )
        # Validation data (strips): red
        validation_mask = high_density_mask & in_strip
        plt.scatter(
            tsne1[validation_mask],
            tsne2[validation_mask],
            c="red",
            s=1,
            label="Validation Data",
            alpha=0.7,
        )
        plt.title("PCA + t-SNE of Flattened Species Trajectories (Log Density >= 6.7)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            f"{self.plots_dir}/pca_tsne_high_density_excluding_strips.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("Three plots generated.")

    def plot_training_val(self):
        """Generate training vs validation t-SNE plot.

        - Uses the same data and filtering as pca_tsne_high_density_excluding_strips.png
        - Loads pre-computed t-SNE data for all high-density models
        - Separates training (non-strips) and validation (strips) data
        - Creates overlay plot identical to pca_tsne_high_density_excluding_strips.png
        - Saves high-resolution plot to plots directory
        """
        print("Generating training vs validation t-SNE plot...")

        # Load the same t-SNE data used in plot_three
        data = self.load_tsne_data(self.tsne_data_path)
        tsne1 = data[:, 1]
        tsne2 = data[:, 2]
        log_densities = data[:, 3]

        # Apply same filters as plot_three
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

        high_density_mask = log_densities >= 6.7
        training_mask = high_density_mask & ~in_strip
        validation_mask = high_density_mask & in_strip

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
            f"{self.plots_dir}/tsne_training_vs_validation.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print("Training vs validation t-SNE plot generated.")


def main():
    """Main function to handle command line arguments.

    - Parses command line arguments using argparse
    - Initializes TSNETool instance
    - Executes the requested operation mode
    - Supports all available t-SNE processing and plotting modes
    """
    parser = argparse.ArgumentParser(description="Unified t-SNE Tool for AstroChemNet")
    parser.add_argument(
        "mode",
        choices=[
            "save-species-tsne",
            "save-phys-tsne",
            "save-indices",
            "plot-filtered",
            "plot-phys",
            "plot-three",
            "plot-training-val",
        ],
        help="Operation mode",
    )

    args = parser.parse_args()
    tool = TSNETool()

    mode_functions = {
        "save-species-tsne": tool.save_species_tsne,
        "save-phys-tsne": tool.save_phys_tsne,
        "save-indices": tool.save_indices,
        "plot-filtered": tool.plot_filtered,
        "plot-phys": tool.plot_phys,
        "plot-three": tool.plot_three,
        "plot-training-val": tool.plot_training_val,
    }

    mode_functions[args.mode]()


if __name__ == "__main__":
    main()
