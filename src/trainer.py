"""Training infrastructure for autoencoder and emulator models."""

import copy
import gc
import json
import os
from abc import abstractmethod
from datetime import datetime
from typing import Any, Final

import numpy as np
import torch
from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import src.AstroChemNet.data_processing as dp
from src.AstroChemNet.config_schemas import DatasetConfig, ModelsConfig
from src.AstroChemNet.loss import Loss

from .models.autoencoder import Autoencoder
from .models.emulator import Emulator

# Performance optimizations
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(True)

# Training constants
MAX_EPOCHS: Final[int] = 9_999_999


class Trainer:
    """Base trainer class with early stopping, adaptive dropout, and learning rate scheduling."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelsConfig,
        model: Autoencoder | Emulator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        device: str,
    ) -> None:
        """Initializes the Trainer class. A class which simplifies training by including all necessary components."""
        self.device = device
        self.start_time = datetime.now()
        self.dataset_config = dataset_config
        self.model_config = model_config

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.num_validation_elements = len(self.validation_dataloader.dataset)  # type: ignore[arg-type]

        self.current_dropout_rate = self.model_config.dropout
        self.current_learning_rate = self.model_config.lr
        self.best_weights: dict[str, Any] | None = None
        self.metric_minimum_loss: float = float("inf")
        self.epoch_validation_loss = torch.zeros(dataset_config.num_species).to(
            self.device
        )
        self.stagnant_epochs = 0
        self.loss_per_epoch = []

    def save_loss_per_epoch(self) -> None:
        """Save loss per epoch to JSON file."""
        epochs_path = os.path.splitext(self.model_config.save_model_path)[0] + ".json"
        with open(epochs_path, "w") as f:
            json.dump(self.loss_per_epoch, f, indent=4)

    def print_final_time(self) -> None:
        """Print total training time and epoch count."""
        end_time = datetime.now()
        total_time = end_time - self.start_time
        print(f"Total Training Time: {total_time}")
        print(f"Total Epochs: {len(self.loss_per_epoch)}")

    def _save_checkpoint(self) -> None:
        """Save model checkpoint to disk."""
        if self.model_config.save_model:
            torch.save(self.model.state_dict(), self.model_config.save_model_path)

    def set_dropout_rate(self, dropout_rate: float) -> None:
        """Update dropout rate for all dropout layers in the model."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate
        self.current_dropout_rate = dropout_rate

    def _cleanup_memory(self) -> None:
        """Release GPU memory and run garbage collection."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def _check_early_stopping(self) -> bool:
        """Check if training should stop due to stagnant progress."""
        if self.stagnant_epochs >= self.model_config.stagnant_epoch_patience:
            print("Ending training early due to stagnant epochs.")
            return True
        return False

    def _check_minimum_loss(self) -> None:
        """Track and update minimum validation loss with adaptive training strategies."""
        val_loss = self.epoch_validation_loss / self.num_validation_elements
        mean_loss = val_loss.mean().item()
        std_loss = val_loss.std().item()
        max_loss = val_loss.max().item()
        metric = mean_loss

        if metric < self.metric_minimum_loss:
            pct_improvement = 100 - metric * 100 / self.metric_minimum_loss
            print("**********************")
            print(
                f"New Minimum\n"
                f"Mean: {mean_loss:.3e}\n"
                f"Std: {std_loss:.3e}\n"
                f"Max: {max_loss:.3e}\n"
                f"Metric: {metric:.3e}\n"
                f"Percent Improvement: {pct_improvement:.3f}%"
            )
            self._save_checkpoint()
            self.best_weights = copy.deepcopy(self.model.state_dict())

            self.metric_minimum_loss = metric
            self.stagnant_epochs = 0
        else:
            self.stagnant_epochs += 1
            print(
                f"Stagnant {self.stagnant_epochs}\n"
                f"Minimum: {self.metric_minimum_loss:.3e}\n"
                f"Mean: {mean_loss:.3e}\n"
                f"Std: {std_loss:.3e}\n"
                f"Max: {max_loss:.3e}\n"
                f"Metric: {metric:.3e}"
            )

            if self.stagnant_epochs % self.model_config.dropout_decay_patience == 0:
                new_dropout = max(
                    self.current_dropout_rate
                    - self.model_config.dropout_reduction_factor,
                    0.0,
                )
                if new_dropout != self.current_dropout_rate:
                    self.stagnant_epochs = 0
                    self.set_dropout_rate(new_dropout)

                    self.current_learning_rate = (
                        1e-3 if new_dropout <= 0.1 else self.model_config.lr
                    )

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.current_learning_rate

                    print(
                        f"Decreasing dropout rate to {self.current_dropout_rate:.4f} "
                        f"and setting lr to {self.current_learning_rate:.4f}."
                    )

            if self.stagnant_epochs == self.model_config.lr_decay_patience + 1:
                if self.best_weights is not None:
                    print("Reverting to previous best weights")
                    self.model.load_state_dict(self.best_weights)

        self.loss_per_epoch.append(
            {
                "mean": mean_loss,
                "std": std_loss,
                "max": max_loss,
                "metric": metric,
                "dropout": self.current_dropout_rate,
                "learning_rate": self.current_learning_rate,
            }
        )
        self.epoch_validation_loss.zero_()
        self.scheduler.step(metric)
        print()
        print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.3e}")
        print(f"Current Dropout Rate: {self.current_dropout_rate:.4f}")
        print(f"Current Num Epochs: {len(self.loss_per_epoch)}")

    @abstractmethod
    def _run_epoch(self, epoch: int) -> None:
        """Execute one training+validation epoch (must be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _run_epoch")

    def train(self) -> None:
        """Main training loop with early stopping."""
        self._cleanup_memory()
        if self.device == "cuda":
            torch.cuda.synchronize()

        for epoch in range(MAX_EPOCHS):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break

        self._cleanup_memory()
        print(f"\nTraining Complete. Final Loss: {self.metric_minimum_loss}")
        self.print_final_time()
        self.save_loss_per_epoch()


class AutoencoderTrainer(Trainer):
    """Trainer specialized for autoencoder models with reconstruction loss."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        model_config: ModelsConfig,
        loss_functions: Loss,
        autoencoder: Autoencoder,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        device: str,
    ) -> None:
        """Initialize AutoencoderTrainer with autoencoder-specific configuration."""
        self.gradient_clipping = model_config.gradient_clipping
        self.ae = autoencoder
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation

        super().__init__(
            dataset_config,
            model_config=model_config,
            model=autoencoder,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            device=device,
        )

    def _run_training_batch(self, features: torch.Tensor) -> None:
        """Execute single training batch (autoencoder reconstruction)."""
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.training_loss(outputs, features)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(self, features: torch.Tensor) -> None:
        """Execute single validation batch (autoencoder reconstruction)."""
        component_outputs = self.ae.encode(features)
        outputs = self.ae.decode(component_outputs)

        loss = self.validation_loss(outputs, features)
        self.epoch_validation_loss += loss

    def _run_epoch(self, epoch: int) -> None:
        """Execute one epoch of autoencoder training and validation."""
        self.training_dataloader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]

        tic1 = datetime.now()
        self.model.train()
        for features in self.training_dataloader:
            features = features[0].to(self.device, non_blocking=True)
            self._run_training_batch(features)

        tic2 = datetime.now()
        self.model.eval()
        with torch.no_grad():
            for features in self.validation_dataloader:
                features = features[0].to(self.device, non_blocking=True)
                self._run_validation_batch(features)

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}\n")


class EmulatorTrainerSequential(Trainer):
    """Trainer specialized for sequential emulator models with autoregressive rollout."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        autoencoder_config: ModelsConfig,
        emulator_config: ModelsConfig,
        loss_functions: Loss,
        processing_functions: dp.Processing,
        autoencoder: Autoencoder,
        emulator: Emulator,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        device: str,
    ) -> None:
        """Initialize EmulatorTrainer with emulator-specific configuration."""
        self.ae = autoencoder
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation
        self.latent_dim = autoencoder_config.latent_dim
        self.gradient_clipping = emulator_config.gradient_clipping
        self.inverse_latent_components_scaling = (
            processing_functions.inverse_latent_components_scaling
        )

        super().__init__(
            dataset_config,
            model_config=emulator_config,
            model=emulator,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
            device=device,
        )

    def _run_training_batch(
        self, phys: torch.Tensor, features: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Execute single emulator training batch."""
        self.optimizer.zero_grad()

        outputs = self.model(phys, features)

        outputs = outputs.reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
        targets = targets.reshape(-1, self.dataset_config.num_species)

        loss = self.training_loss(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(
        self, phys: torch.Tensor, features: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Execute single emulator validation batch."""
        outputs = self.model(phys, features)

        outputs = outputs.reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
        outputs = outputs.reshape(targets.size(0), targets.size(1), -1)

        loss = self.validation_loss(outputs, targets).mean(dim=0)

        self.epoch_validation_loss += loss.detach()

    def _run_epoch(self, epoch: int) -> None:
        """Execute one epoch of emulator training and validation."""
        self.training_dataloader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        tic1 = datetime.now()

        self.model.train()

        for phys, features, targets in self.training_dataloader:
            phys = phys.to(self.device, non_blocking=True)
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self._run_training_batch(phys, features, targets)

        tic2 = datetime.now()

        self.model.eval()
        with torch.no_grad():
            for phys, features, targets in self.validation_dataloader:
                phys = phys.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self._run_validation_batch(phys, features, targets)

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}")


def load_objects(
    model: Autoencoder | Emulator, config: ModelsConfig
) -> tuple[optim.AdamW, ReduceLROnPlateau]:
    """Create optimizer and scheduler for model training."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=tuple(config.betas),  # type: ignore[arg-type]
        weight_decay=config.weight_decay,
        fused=True,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_decay,
        patience=config.lr_decay_patience,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    return optimizer, scheduler
