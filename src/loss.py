"""Loss functions for autoencoder and emulator training."""

from typing import Optional, Tuple

import numpy as np
import torch

from configs.autoencoder import AEConfig
from configs.emulator import EMConfig
from configs.general import GeneralConfig


class Loss:
    """Loss functions for training autoencoder and emulator models."""

    def __init__(
        self,
        processing_functions,
        GeneralConfig: GeneralConfig,
        ModelConfig: Optional[AEConfig | EMConfig] = None,
    ) -> None:
        """Initialize Loss with processing functions and configuration."""
        device = GeneralConfig.device
        stoichiometric_matrix = np.load(GeneralConfig.stoichiometric_matrix_path)
        self.stoichiometric_matrix = torch.tensor(
            stoichiometric_matrix, dtype=torch.float32, device=device
        )
        self.exponential = torch.log(torch.tensor(10, device=device).float())

        self.inverse_abundances_scaling = (
            processing_functions.inverse_abundances_scaling
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate elementwise exponential loss with mean and worst case."""
        elementwise_loss = torch.abs(outputs - targets)
        elementwise_loss = torch.exp(power_weight * exponential * elementwise_loss) - 1

        mean_loss = torch.sum(elementwise_loss) / targets.size(0)

        worst_loss = elementwise_loss.max(dim=1).values.mean()

        return mean_loss, worst_loss

    def elemental_conservation(
        self,
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate loss between elemental abundances of actual and predicted."""
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
    ) -> torch.Tensor:
        """Calculate combined training loss with reconstruction and conservation."""
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

    def validation(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate validation loss as relative error per species."""
        unscaled_outputs = self.inverse_abundances_scaling(outputs)
        unscaled_targets = self.inverse_abundances_scaling(targets)

        loss = torch.abs(unscaled_targets - unscaled_outputs) / unscaled_targets

        return torch.sum(loss, dim=0)
