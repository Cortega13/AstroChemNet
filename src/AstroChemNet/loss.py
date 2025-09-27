import torch
import numpy as np


class Loss:
    def __init__(self, processing_functions, GeneralConfig, ModelConfig=None):
        device = GeneralConfig.device
        stoichiometric_matrix = np.load(GeneralConfig.stoichiometric_matrix_path)
        molecular_matrix = np.load(GeneralConfig.molecular_matrix_path)
        self.stoichiometric_matrix = torch.tensor(
            stoichiometric_matrix, dtype=torch.float32, device=device
        )
        self.molecular_matrix = torch.tensor(
            molecular_matrix, dtype=torch.float32, device=device
        )
        atomic_species = ["H", "HE", "C", "N", "O", "S", "SI", "MG", "CL"]
        self.atomic_indices = torch.tensor(
            [GeneralConfig.species.index(sp) for sp in atomic_species],
            dtype=torch.long,
            device=device,
        )
        self.elemental_indices = torch.arange(len(atomic_species), device=device)

        elemental_abundances = GeneralConfig.initial_abundances @ stoichiometric_matrix
        self.elemental_abundances = torch.tensor(
            elemental_abundances, dtype=torch.float32, device=device
        )

        self.exponential = torch.log(torch.tensor(10, device=device).float())

        self.inverse_abundances_scaling = (
            processing_functions.inverse_abundances_scaling
        )
        self.abundances_scaling = processing_functions.jit_abundances_scaling

        if ModelConfig:
            self.power_weight = torch.tensor(
                ModelConfig.power_weight, dtype=torch.float32, device=device
            )
            self.conservation_weight = torch.tensor(
                ModelConfig.conservation_weight, dtype=torch.float32, device=device
            )

    @staticmethod
    def elementwise_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        exponential: torch.Tensor,
        power_weight: torch.Tensor,
    ):
        elementwise_loss = torch.abs(outputs - targets)
        elementwise_loss = torch.exp(power_weight * exponential * elementwise_loss) - 1

        mean_loss = torch.sum(elementwise_loss) / targets.size(0)

        worst_loss = elementwise_loss.max(dim=1).values.mean()

        return mean_loss, worst_loss

    def elemental_conservation(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ):
        """
        Given the actual and predicted abundances, this function calculates a loss between the elemental abundances of both.
        """
        unscaled_tensor1 = self.inverse_abundances_scaling(tensor1)
        unscaled_tensor2 = self.inverse_abundances_scaling(tensor2)

        elemental_abundances1 = torch.abs(
            torch.matmul(unscaled_tensor1, self.stoichiometric_matrix)
        )
        elemental_abundances2 = torch.abs(
            torch.matmul(unscaled_tensor2, self.stoichiometric_matrix)
        )

        log_elemental_abundances1 = torch.log10(elemental_abundances1)
        log_elemental_abundances2 = torch.log10(elemental_abundances2)

        diff = torch.abs(log_elemental_abundances2 - log_elemental_abundances1)

        return torch.sum(diff) / tensor1.size(0)

    def _emulator_outputs_reconstruction(self, output: torch.Tensor):
        unscaled_output = self.inverse_abundances_scaling(output)

        atoms = self.elemental_abundances - (unscaled_output @ self.molecular_matrix)

        unscaled_output = unscaled_output.index_copy(
            1, self.atomic_indices, atoms[:, self.elemental_indices]
        )

        unscaled_output = torch.clamp(unscaled_output, min=1e-20)

        reconstructed_output = self.abundances_scaling(unscaled_output)
        return reconstructed_output

    def training(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 2,
        emulator: bool = False,
    ):
        """
        This is the custom loss function for the autoencoder.
        It's a combination of the reconstruction loss, conservation loss,
        and a penalty on the worst-performing species.
        """
        if emulator:
            outputs = self._emulator_outputs_reconstruction(outputs)

        mean_loss, worst_loss = self.elementwise_loss(
            outputs, targets, self.exponential, self.power_weight
        )

        conservation_error = self.conservation_weight * self.elemental_conservation(
            outputs, targets
        )

        # combine everything
        total_loss = 1e-3 * (mean_loss + alpha * worst_loss)

        print(
            f"Recon: {mean_loss.detach():.3e} | Worst: {worst_loss.detach():.3e} "
            f"| Cons: {conservation_error.detach():.3e} | Total: {total_loss.detach():.3e}"
        )
        return total_loss

    def validation(self, outputs, targets):
        """
        This is the custom loss function for the autoencoder. It's a combination of the reconstruction loss and the conservation loss.
        """
        unscaled_outputs = self.inverse_abundances_scaling(outputs)
        unscaled_targets = self.inverse_abundances_scaling(targets)

        loss = torch.abs(unscaled_targets - unscaled_outputs) / unscaled_targets

        return torch.sum(loss, dim=0)
