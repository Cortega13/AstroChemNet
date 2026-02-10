"""Loss functions for training."""

import numpy as np
import torch


class Loss:
    """Loss functions for training."""

    def __init__(self, processing_functions, dataset_cfg, model_cfg=None):
        """Initialize loss functions with processing and configuration objects."""
        device = processing_functions.device
        stoichiometric_matrix = torch.load(
            dataset_cfg.stoichiometric_matrix_path, map_location="cpu"
        )
        if isinstance(stoichiometric_matrix, np.ndarray):
            stoichiometric_matrix = torch.from_numpy(stoichiometric_matrix)
        self.stoichiometric_matrix = stoichiometric_matrix.to(
            dtype=torch.float32, device=device
        )
        self.exponential = torch.log(torch.tensor(10, device=device).float())

        self.inverse_abundances_scaling = (
            processing_functions.inverse_abundances_scaling
        )

        if model_cfg:
            self.power_weight = torch.tensor(
                model_cfg.power_weight, dtype=torch.float32, device=device
            )
            self.conservation_weight = torch.tensor(
                model_cfg.conservation_weight, dtype=torch.float32, device=device
            )

    @staticmethod
    def elementwise_loss(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        exponential: torch.Tensor,
        power_weight: torch.Tensor,
    ):
        """Calculate elementwise loss."""
        elementwise_loss = torch.abs(outputs - targets)

        mean_loss = torch.sum(elementwise_loss) / targets.size(0)

        worst_loss = elementwise_loss.max(dim=1).values.mean()

        return mean_loss, worst_loss

    def elemental_conservation(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ):
        """Given the actual and predicted abundances, this function calculates a loss between the elemental abundances of both."""
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
        alpha: float = 2,
    ):
        """Custom loss function for the autoencoder."""
        mean_loss, worst_loss = self.elementwise_loss(
            outputs, targets, self.exponential, self.power_weight
        )

        conservation_error = self.conservation_weight * self.elemental_conservation(
            outputs, targets
        )

        # combine everything
        total_loss = 1e-3 * (mean_loss + alpha * worst_loss + conservation_error)

        print(
            f"Recon: {mean_loss.detach():.3e} | Worst: {worst_loss.detach():.3e} "
            f"| Cons: {conservation_error.detach():.3e} | Total: {total_loss.detach():.3e}"
        )
        return total_loss

    def validation(self, outputs, targets):
        """This is the custom loss function for the autoencoder. It's a combination of the reconstruction loss and the conservation loss."""
        unscaled_outputs = self.inverse_abundances_scaling(outputs)
        unscaled_targets = self.inverse_abundances_scaling(targets)

        loss = torch.abs(unscaled_targets - unscaled_outputs) / unscaled_targets

        return torch.sum(loss, dim=0)
