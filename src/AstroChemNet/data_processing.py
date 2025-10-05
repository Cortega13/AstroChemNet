"""
We grouped all the data processing functions into a single class for better organization and reusability.
We have the preprocessing and postprocessing functions. These include scaling the abundances and physical parameters.
"""

import numpy as np
import torch
import re
import gc
from numba import njit
from .inference import Inference


class Processing:
    """
    Just for grouping functions with a similar purpose. That way they all have access to tensors which are loaded during configuration.
    """

    def __init__(self, GeneralConfig, AEConfig=None):
        self.device = GeneralConfig.device
        self.exponential = torch.log(torch.tensor(10, device=self.device).float())

        self.abundances_min = torch.tensor(
            np.log10(GeneralConfig.abundances_lower_clipping),
            dtype=torch.float32,
            device=self.device,
        )
        self.abundances_max = torch.tensor(
            np.log10(GeneralConfig.abundances_upper_clipping),
            dtype=torch.float32,
            device=self.device,
        )
        self.abundances_min_np = self.abundances_min.cpu().numpy()
        self.abundances_max_np = self.abundances_max.cpu().numpy()

        if AEConfig is not None:
            latents_minmax = np.load(AEConfig.latents_minmax_path)
            print(f"Latents MinMax: {latents_minmax[0]}, {latents_minmax[1]}")
            self.components_min = torch.tensor(
                latents_minmax[0], dtype=torch.float32, device=self.device
            )
            self.components_max = torch.tensor(
                latents_minmax[1], dtype=torch.float32, device=self.device
            )

        self.physical_parameter_ranges = GeneralConfig.physical_parameter_ranges
        self.species = GeneralConfig.species
        self.num_species = GeneralConfig.num_species
        self.stoichiometric_matrix_path = GeneralConfig.stoichiometric_matrix_path

    ### PreProcessing Functions

    def physical_parameter_scaling(self, physical_parameters: np.ndarray):
        """
        Preprocesses the dataset by minmax scaling the physical parameters to be within [0, 1].
        """
        np.log10(physical_parameters, out=physical_parameters)
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)

            physical_parameters[:, i] = (physical_parameters[:, i] - log_param_min) / (
                log_param_max - log_param_min
            )

    def jit_abundances_scaling(
        self,
        abundances: torch.Tensor,
    ):
        """
        Abundances are log10'd and then minmax scaled to be within [0, 1].
        """
        abundances = torch.log10(abundances)
        abundances = (abundances - self.abundances_min) / (
            self.abundances_max - self.abundances_min
        )
        return abundances

    def abundances_scaling(
        self,
        abundances: np.ndarray,
    ):
        """
        Abundances are log10'd and then minmax scaled to be within [0, 1].
        """
        np.log10(abundances, out=abundances)
        np.subtract(abundances, self.abundances_min_np, out=abundances)
        np.divide(
            abundances,
            (self.abundances_max_np - self.abundances_min_np),
            out=abundances,
        )

    def latent_components_scaling(
        self,
        components: torch.Tensor,
    ):
        """
        Latent components are scaled to be within [0, 1].
        """

        return (components - self.components_min) / (
            self.components_max - self.components_min
        )

    ### PostProcessing Functions

    def inverse_physical_parameter_scaling(self, physical_parameters: np.array):
        """
        Reverses the minmax scaling of the physical parameters.
        Operates in-place.
        """
        for i, parameter in enumerate(self.physical_parameter_ranges):
            param_min, param_max = self.physical_parameter_ranges[parameter]
            log_param_min, log_param_max = np.log10(param_min), np.log10(param_max)

            physical_parameters[:, i] = (
                physical_parameters[:, i] * (log_param_max - log_param_min)
                + log_param_min
            )

        np.power(10, physical_parameters, out=physical_parameters)

    @staticmethod
    @torch.jit.script
    def jit_inverse_abundances_scaling(
        abundances: torch.Tensor,
        min_: torch.Tensor,
        max_: torch.Tensor,
        exponential_: torch.Tensor,
    ):
        """
        We use jit to compile the function so that during training there is no casting between array types and cpu/gpu.

        Reverses the minmax scaling of the abundances.
        """
        log_abundances = abundances * (max_ - min_) + min_
        abundances = torch.exp(exponential_ * log_abundances)
        return abundances

    def inverse_abundances_scaling(self, abundances):
        """
        Reverses the minmax scaling of the abundances.

        Is able to handle both torch tensor or numpy inputs.
        """
        if isinstance(abundances, torch.Tensor):
            abundances = self.jit_inverse_abundances_scaling(
                abundances,
                self.abundances_min,
                self.abundances_max,
                self.exponential,
            )
            return abundances
        else:
            ab_min_np = self.abundances_min.cpu().numpy()
            ab_max_np = self.abundances_max.cpu().numpy()
            exponential_np = self.exponential.cpu().numpy()

            np.multiply(abundances, (ab_max_np - ab_min_np), out=abundances)
            np.add(abundances, ab_min_np, out=abundances)
            np.exp(exponential_np * abundances, out=abundances)

    @staticmethod
    @torch.jit.script
    def jit_inverse_latent_component_scaling(
        scaled_components: torch.Tensor, min_: torch.Tensor, max_: torch.Tensor
    ):
        """
        We use jit to compile the function so that during training there is no casting between array types and cpu/gpu.

        Reverses the latent component scaling.
        """
        return scaled_components * (max_ - min_) + min_

    def inverse_latent_components_scaling(
        self,
        scaled_components: torch.Tensor,
    ):
        """
        Reverses the latent component scaling.
        """
        return self.jit_inverse_latent_component_scaling(
            scaled_components, self.components_min, self.components_max
        )

    def save_latents_minmax(
        self,
        AEConfig,
        dataset_t: torch.Tensor,
        inference_functions: Inference,
    ):
        """
        For minmax scaling the latent components we need to have the minimum and maximum latent values produced by the autoencoder.

        This function saves them to the path defined in the general configuration file.
        """
        min_, max_ = float("inf"), float("-inf")

        with torch.no_grad():
            for i in range(0, len(dataset_t), AEConfig.batch_size):
                batch = dataset_t[i : i + AEConfig.batch_size].to(self.device)
                encoded = inference_functions.encode(batch).cpu()
                min_ = min(min_, encoded.min().item())
                max_ = max(max_, encoded.max().item())

        minmax_np = np.array([min_, max_], dtype=np.float32)
        print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")
        np.save(AEConfig.latents_minmax_path, minmax_np)

    def build_stoichiometric_matrix(self):
        """
        Generates a stoichiometric matrix S from the species x in the dataset.

        By doing the operation x @ S we obtain the elemental abundances n.

        Returns a numpy matrix of shape (num_species, num_elements).

        (Currently does not account for electrons, since they are for whatever reason not conserved by UCLCHEM. Still debugging this.)
        (Currently does not include SURFACE or BULK since we saw no improvements for conserving them.)
        """
        elements = ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]
        stoichiometric_matrix = np.zeros((len(elements), self.num_species))
        modified_species = [s.replace("@", "").replace("#", "") for s in self.species]

        elements_patterns = {
            "H": re.compile(r"H(?!E)(\d*)"),
            "HE": re.compile(r"HE(\d*)"),
            "C": re.compile(r"C(?!L)(\d*)"),
            "N": re.compile(r"N(\d*)"),
            "O": re.compile(r"O(\d*)"),
            "S": re.compile(r"S(?!I)(\d*)"),
            "SI": re.compile(r"SI(\d*)"),
            "MG": re.compile(r"MG(\d*)"),
            "CL": re.compile(r"CL(\d*)"),
        }

        for element, pattern in elements_patterns.items():
            elem_index = elements.index(element)
            for i, species in enumerate(modified_species):
                match = pattern.search(species)
                if match and species not in ["SURFACE", "BULK"]:
                    multiplier = int(match.group(1)) if match.group(1) else 1
                    stoichiometric_matrix[elem_index, i] = multiplier

        np.save(self.stoichiometric_matrix_path, stoichiometric_matrix.T)
        return stoichiometric_matrix.T


@njit
def calculate_emulator_indices(
    dataset_np: np.ndarray,
    window_size: int = 16,
):
    """
    The emulator training elements have significant overlap in the rows they use from the dataset.
    This basically generates which indices are needed for each training element, so that during training it is recalled on the fly.
    """
    change_indices = np.where(np.diff(dataset_np[:, 1].astype(np.int32)) != 0)[0] + 1
    model_groups = np.split(dataset_np, change_indices)

    total_seqs = 0
    for group in model_groups:
        n = len(group)
        total_seqs += n - window_size + 1

    sequences = np.full((total_seqs, window_size), -1, dtype=np.int32)

    seq_idx = 0
    for group in model_groups:
        indices = group[:, 0]
        n = len(indices)
        for start_idx in range(n - window_size + 1):
            sequences[seq_idx, :] = indices[start_idx : start_idx + window_size]
            seq_idx += 1

    return sequences


def preprocessing_emulator_dataset(
    GeneralConfig,
    EMConfig,
    dataset_np: np.array,
    processing_functions: Processing,
    inference_functions: Inference,
):
    """
    Generates index pairs for training.
    Generates latent components using autoencoder for the dataset.
    Scales physical parameters
    """
    num_species = GeneralConfig.num_species
    num_phys = GeneralConfig.num_phys
    num_metadata = GeneralConfig.num_metadata

    dataset_np[:, 0] = np.arange(len(dataset_np))

    processing_functions.physical_parameter_scaling(
        dataset_np[:, num_metadata : num_metadata + num_phys]
    )
    processing_functions.abundances_scaling(dataset_np[:, -num_species:])

    latent_components = inference_functions.encode(
        dataset_np[:, num_metadata + num_phys :]
    )
    latent_components = (
        processing_functions.latent_components_scaling(latent_components).cpu().numpy()
    )
    encoded_dataset_np = np.hstack((dataset_np, latent_components), dtype=np.float32)

    index_pairs_np = calculate_emulator_indices(
        encoded_dataset_np, EMConfig.window_size
    )

    perm = np.random.permutation(len(index_pairs_np))
    index_pairs_shuffled_np = index_pairs_np[perm]

    encoded_t = torch.from_numpy(encoded_dataset_np).float()
    index_pairs_shuffled_t = torch.from_numpy(index_pairs_shuffled_np).int()

    gc.collect()
    torch.cuda.empty_cache()

    return (encoded_t, index_pairs_shuffled_t)
