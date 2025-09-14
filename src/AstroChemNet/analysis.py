import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
from .configs import DatasetConfig, PredefinedTensors
from . import data_processing as dp


def sample_initial_conditions(
    n_samples: int = 100000,
    convert_base_av: bool = True,
):
    """
    Generate initial conditions for the parameter sampling plots where physical conditions are varied.
    """
    inputs = np.random.uniform(
        0, 1, size=(n_samples, DatasetConfig.num_physical_parameters)
    )
    if convert_base_av:
        dp.inverse_physical_parameter_scaling(inputs)
        dp.baseAvtoAv(inputs)
        dp.physical_parameter_scaling(inputs)

    initial_abundances = DatasetConfig.initial_abundances.copy()
    dp.abundances_scaling(initial_abundances)

    initial_abundances_repeated = np.tile(initial_abundances, (n_samples, 1))
    inputs = np.hstack((inputs, initial_abundances_repeated))
    return inputs


def add_timesteps_to_conditions(initial_conditions: np.array, num_timesteps: int = 1):
    """
    Adds a time column to the initial conditions tensor.
    """
    time_as_fraction = num_timesteps / DatasetConfig.num_timesteps_per_model
    batch_size = initial_conditions.shape[0]
    time_column = np.full((batch_size, 1), time_as_fraction)

    initial_conditions_with_time = np.hstack((time_column, initial_conditions))
    return initial_conditions_with_time


def add_multiple_timesteps_to_conditions(
    initial_conditions: torch.Tensor, num_timesteps: int = 95
):
    """
    Expands the initial conditions tensor to include copies for each timestep.
    Adds a time column to the expanded tensor.
    """
    time_values = (
        torch.linspace(1, num_timesteps, num_timesteps).reshape(num_timesteps, 1) / 100
    )

    expanded_conditions = initial_conditions.repeat(num_timesteps, 1)

    conditions_with_time = torch.cat((time_values, expanded_conditions), dim=1)

    return conditions_with_time


def reconstruct_results(inputs, outputs):
    physical_parameters = inputs[:, 1 : 1 + DatasetConfig.num_physical_parameters]
    dp.inverse_physical_parameter_scaling(physical_parameters)

    columns = DatasetConfig.physical_parameters + DatasetConfig.species
    results = np.hstack((physical_parameters, outputs))
    results_df = pd.DataFrame(results, columns=columns)

    return results_df


def benchmark_speed(DATASET, AE_CONFIG, EMULATOR_CONFIG):
    pass


### Plot Functions
def histogram_physical_parameters(
    sampled_physical_parameters: np.array, savefig_path: str = None
):
    """
    Generate histograms for the sampled physical conditions to visualize distribution.
    """

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, param in enumerate(DatasetConfig.physical_parameters):
        row = i // 2
        col = i % 2
        axs[row, col].hist(
            np.log10(sampled_physical_parameters[:, i]),
            bins=200,
            color="steelblue",
            edgecolor="black",
        )
        axs[row, col].set_title(f"{param} Frequency")
        axs[row, col].set_xlabel("Value")
        axs[row, col].set_ylabel("Count")

    plt.tight_layout()
    if savefig_path:
        plt.savefig(savefig_path, dpi=300, bbox_inches="tight")


def plot_abundances_vs_time_comparison(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    species_of_interest: list,
    output_path: str = None,
):
    """
    Plotting the reconstructed and original abundances on a chemical evolution plot.
    This shows how accurate the reconstructed evolution is.
    """
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


def scatter_abundances_vs_physical_parameters(
    training_df: pd.DataFrame,
    inference_df: pd.DataFrame,
    species_of_interest: list,
    output_folder: str = "plots/compare_scatter_abundances",
):
    """
    Generates comparative scatter plots of training vs. predicted abundances
    for each species vs. each physical parameter.
    """

    combined_df = pd.concat([training_df, inference_df], ignore_index=True)
    global_mins = np.log10(combined_df[DatasetConfig.physical_parameters].min())
    global_maxs = np.log10(combined_df[DatasetConfig.physical_parameters].max())

    for species in species_of_interest:
        fig, axes = plt.subplots(4, 2, figsize=(12, 16))

        for i, varying_param in enumerate(DatasetConfig.physical_parameters):
            df_subset = training_df.sort_values(by=varying_param, ascending=True)
            other_params = [
                p for p in DatasetConfig.physical_parameters if p != varying_param
            ]
            df_color = np.log10(df_subset[other_params].astype(float))
            colors = (df_color - global_mins[other_params]) / (
                global_maxs[other_params] - global_mins[other_params]
            )
            colors = (colors - colors.min()) / (colors.max() - colors.min())
            colors = 1 / (1 + np.exp(-10 * (colors - 0.5)))
            colors *= 0.8
            colors = colors.to_numpy()

            ax_train = axes[i, 0]
            ax_train.scatter(
                np.log10(df_subset[varying_param]),
                np.log10(df_subset[species]),
                c=colors,
                marker=".",
                linewidth=0.1,
            )
            ax_train.set_xlabel(f"Log {varying_param}")
            ax_train.set_ylabel(f"Log {species} Abundance")
            ax_train.set_title(
                f"Training: Log {species} Abundance vs. Log {varying_param}"
            )
            ax_train.grid(True)

            channel_info = (
                f"R = {other_params[0]} "
                f"[{global_mins[other_params[0]]:.2e}, {global_maxs[other_params[0]]:.2e}]\n"
                f"G = {other_params[1]} "
                f"[{global_mins[other_params[1]]:.2e}, {global_maxs[other_params[1]]:.2e}]\n"
                f"B = {other_params[2]} "
                f"[{global_mins[other_params[2]]:.2e}, {global_maxs[other_params[2]]:.2e}]"
            )
            ax_train.text(
                0.0,
                1,
                channel_info,
                transform=ax_train.transAxes,
                va="top",
                ha="left",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
            )

            df_subset = inference_df.sort_values(by=varying_param, ascending=True)
            df_color = np.log10(df_subset[other_params].astype(float))
            colors = (df_color - global_mins[other_params]) / (
                global_maxs[other_params] - global_mins[other_params]
            )
            colors = (colors - colors.min()) / (colors.max() - colors.min())
            colors = 1 / (1 + np.exp(-10 * (colors - 0.5)))
            colors *= 0.8
            colors = colors.to_numpy()

            ax_pred = axes[i, 1]
            ax_pred.scatter(
                np.log10(df_subset[varying_param]),
                np.log10(df_subset[species]),
                c=colors,
                marker=".",
                linewidth=0.1,
            )
            ax_pred.set_xlabel(f"Log {varying_param}")
            ax_pred.set_ylabel(f"Log {species} Abundance")
            ax_pred.set_title(
                f"Predicted: Log {species} Abundance vs. Log {varying_param}"
            )
            ax_pred.grid(True)

            ax_pred.text(
                0.0,
                1,
                channel_info,
                transform=ax_pred.transAxes,
                va="top",
                ha="left",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"),
            )

            y_min = min(
                np.log10(training_df[species].min()),
                np.log10(inference_df[species].min()),
            )
            y_max = max(
                np.log10(training_df[species].max()),
                np.log10(inference_df[species].max()),
            )
            x_min = min(
                np.log10(training_df[varying_param].min()),
                np.log10(inference_df[varying_param].min()),
            )
            x_max = max(
                np.log10(training_df[varying_param].max()),
                np.log10(inference_df[varying_param].max()),
            )

            ax_train.set_xlim(x_min, x_max)
            ax_train.set_ylim(y_min, y_max)
            ax_pred.set_xlim(x_min, x_max)
            ax_pred.set_ylim(y_min, y_max)

        plt.tight_layout()
        folder_path = os.path.join(DatasetConfig.working_path, output_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        savefig_path = os.path.join(folder_path, f"{species}.png")
        plt.savefig(savefig_path, dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_error_vs_time(
    errors: torch.Tensor,
    label: str = "Mean MRE across species",
    save_path: str = "plots/errors/unnamed.png",
):
    """
    Helper function for plotting errors across time.
    """
    save_path = os.path.join(DatasetConfig.working_path, save_path)

    errors = errors.cpu().numpy()
    timesteps = np.arange(DatasetConfig.num_timesteps_per_model)

    plt.figure(figsize=(7, 5))
    plt.scatter(
        timesteps + 1,
        errors,
        label=f"{label}\nMin: {errors.min():.1f}%\nMax: {errors.max():.1f}%\nMean: {errors.mean():.1f}%",
        color="b",
        zorder=2,
    )
    plt.plot(timesteps + 1, errors, color="b", linewidth=1, alpha=0.7, zorder=1)

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
    return torch.abs(target - feature) / target


@torch.jit.script
def calculate_conservation_error(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    returns
    mean_original_elemental_abundances, mean_reconstruction_elemental_abundances, conservation_error

    """
    unscaled_tensor1 = dp.stoichiometric_matrix_mult(tensor1)
    unscaled_tensor2 = dp.stoichiometric_matrix_mult(tensor2)
    conservation_error = (
        torch.abs(unscaled_tensor1 - unscaled_tensor2) / unscaled_tensor1
    )
    return torch.mean(conservation_error, dim=1)


@torch.jit.script
def calculate_mace_error(
    original_abundances: torch.Tensor,
    reconstructed_abundances: torch.Tensor,
    max_abundance: torch.Tensor = PredefinedTensors.mace_max_abundance,  # Maximum abundance defined in MACE github.
    mace_factor: float = PredefinedTensors.mace_factor,  # MACE has 468 species and we have 335, so we multiply by 468/335.
):
    """
    Calculate the MACE error as defined in the MACE github repository.
    """
    original_clipped = torch.clamp(original_abundances, min=0.0, max=max_abundance)
    reconstructed_clipped = torch.clamp(
        reconstructed_abundances, min=0.0, max=max_abundance
    )

    log_original = torch.log10(original_clipped)
    log_reconstructed = torch.log10(reconstructed_clipped)

    mace_species_error = (log_original - log_reconstructed) / log_reconstructed

    mace_error = torch.abs(mace_species_error) * mace_factor

    return mace_error.sum(dim=1)
