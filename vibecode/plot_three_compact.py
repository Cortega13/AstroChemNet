"""Compact script for plot_three functionality from unified_tsne_tool."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(".."))
import AstroChemNet.data_loading as dl
from configs.autoencoder import AEConfig
from configs.emulator import EMConfig
from configs.general import GeneralConfig


class CompactTSNETool:
    """Compact tool for plot_three t-SNE operations."""

    def __init__(self):
        """Initialize configurations."""
        self.general_config = GeneralConfig()
        self.ae_config = AEConfig()
        self.em_config = EMConfig()
        self.tsne_data_path = "tsne_data.csv"
        self.plots_dir = "../plots/trajectories"
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_datasets(self):
        """Load training and validation datasets."""
        return dl.load_datasets(self.general_config, self.em_config.columns)

    def load_tsne_data(self, path):
        """Load t-SNE data from CSV file."""
        return np.loadtxt(path, delimiter=",", skiprows=1)

    def apply_pca_tsne(self, data, n_components_pca=50, n_components_tsne=2):
        """Apply PCA followed by t-SNE."""
        pca_reduced = PCA(n_components=n_components_pca).fit_transform(data)
        return TSNE(
            n_components=n_components_tsne, random_state=42, perplexity=100
        ).fit_transform(pca_reduced)

    def get_unique_models(self, combined_np):
        """Get unique model IDs."""
        return np.unique(combined_np[:, 1])

    def extract_trajectories(self, combined_np, selected_models, data_type="species"):
        """Extract and process trajectories."""
        trajectories = []
        log_densities = []
        for model in selected_models:
            mask = combined_np[:, 1] == model
            subset = combined_np[mask]
            if data_type == "species":
                data = subset[:, -self.general_config.num_species :].copy()
                processed = np.log10(data + 1e-20).flatten()
            else:
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

    def plot_three(self):
        """Generate three comparison plots."""
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
        plt.figure(figsize=(10, 8))
        training_mask = high_density_mask & ~in_strip
        plt.scatter(
            tsne1[training_mask],
            tsne2[training_mask],
            c="blue",
            s=1,
            label="Training Data",
            alpha=0.7,
        )
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


if __name__ == "__main__":
    tool = CompactTSNETool()
    tool.plot_three()
