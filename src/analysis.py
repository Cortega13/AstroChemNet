"""Analysis Helper Functions."""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import data_processing as dp


def benchmark_speed(DATASET, AE_CONFIG, EMULATOR_CONFIG):
    pass


def plot_abundances_vs_time_comparison(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    species_of_interest: list,
    output_path: str = "",
    DatasetConfig: Any = None,
):
    """Plotting the reconstructed and original abundances on a chemical evolution plot."""
    physical_parameters = actual.loc[:, DatasetConfig.physical_parameters]
    phys_params = physical_parameters.iloc[0]
    params_text = "\n".join(
        [f"{param}: {phys_params[param]:.3f}" for param in physical_parameters.columns]
    )

    plt.figure(figsize=(10, 10))
    colors = plt.colormaps.get_cmap("tab20")

    timesteps = np.arange(0, min(len(actual), len(predicted)))
    for idx, species in enumerate(species_of_interest):
        plt.plot(
            timesteps,
            np.log10(actual[species])[: len(timesteps)],
            label=f"{species} Actual",
            color=colors(idx),
            linestyle="-",
        )
        plt.plot(
            timesteps,
            np.log10(predicted[species])[: len(timesteps)],
            label=f"{species} Predicted",
            color=colors(idx),
            linestyle="--",
        )
    plt.xlabel("Time (x1000 year)")
    plt.ylabel("Log Abundances (Relative to H nuclei)")
    plt.title("Log Abundances vs. Time")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title=params_text)

    plt.tight_layout()
    if output_path:
        output_path = os.path.join(DatasetConfig.working_path, output_path)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")


def plot_error_vs_time(
    errors: torch.Tensor,
    label: str = "Mean MRE across species",
    save_path: str = "plots/errors/unnamed.png",
    DatasetConfig: Any = None,
):
    """Helper function for plotting errors across time."""
    save_path = os.path.join(DatasetConfig.working_path, save_path)

    errors_np = errors.cpu().numpy()
    timesteps = np.arange(DatasetConfig.num_timesteps_per_model)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        timesteps + 1,
        errors_np,
        label=f"{label}\nMin: {errors_np.min():.1f}%\nMax: {errors_np.max():.1f}%\nMean: {errors_np.mean():.1f}%",
        color="b",
        zorder=2,
    )
    plt.plot(timesteps + 1, errors_np, color="b", linewidth=1, alpha=0.7, zorder=1)

    plt.xlabel("Timestep Δt (x1kyr)")
    plt.ylabel(f"{label}")
    plt.title(f"{label} vs. Timestep")
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


### Statistics Functions
@torch.jit.script
def calculate_relative_error(feature, target):
    """Calculates the relative error."""
    return torch.abs(target - feature) / target


@torch.jit.script
def calculate_conservation_error(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """Calculates the conservation error."""
    unscaled_tensor1 = dp.stoichiometric_matrix_mult(tensor1)
    unscaled_tensor2 = dp.stoichiometric_matrix_mult(tensor2)
    conservation_error = (
        torch.abs(unscaled_tensor1 - unscaled_tensor2) / unscaled_tensor1
    )
    return torch.mean(conservation_error, dim=1)
