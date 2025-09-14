import torch
import numpy as np


class Loss:
    def __init__(
        self, processing_functions, GeneralConfig, AEConfig=None, EMConfig=None
    ):
        device = GeneralConfig.device
        stoichiometric_matrix = np.load(GeneralConfig.stoichiometric_matrix_path)
        self.stoichiometric_matrix = torch.tensor(
            stoichiometric_matrix, dtype=torch.float32, device=device
        )
        self.exponential = torch.log(torch.tensor(10, device=device).float())
        self.inverse_abundances_scaling = (
            processing_functions.inverse_abundances_scaling
        )

        if AEConfig is not None:
            self.power_weight = torch.tensor(
                AEConfig.power_weight, dtype=torch.float32, device=device
            )
            self.conservation_weight = torch.tensor(
                AEConfig.conservation_weight, dtype=torch.float32, device=device
            )
        else:
            self.power_weight = torch.tensor(
                EMConfig.power_weight, dtype=torch.float32, device=device
            )
            self.conservation_weight = torch.tensor(
                EMConfig.conservation_weight, dtype=torch.float32, device=device
            )

    @staticmethod
    def elementwise_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        exponential: torch.Tensor,
        power_weight: torch.Tensor,
    ):
        # return ((torch.abs(outputs - targets))**2).sum() / targets.size(0)
        elementwise_loss = torch.abs(outputs - targets)
        elementwise_loss = torch.exp(power_weight * exponential * elementwise_loss) - 1
        elementwise_loss = torch.sum(elementwise_loss) / targets.size(0)
        return elementwise_loss

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

    def training(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        This is the custom loss function for the autoencoder. It's a combination of the reconstruction loss and the conservation loss.
        """

        elementwise_loss = self.elementwise_loss(
            outputs, targets, self.exponential, self.power_weight
        )

        conservation_error = self.conservation_weight * self.elemental_conservation(
            outputs, targets
        )

        total_loss = 1e-3 * (elementwise_loss + conservation_error)

        print(
            f"Recon: {elementwise_loss.detach():.3e} | Cons: {conservation_error.detach():.3e} | Total: {total_loss.detach():.3e}"
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
