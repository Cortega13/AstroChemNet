"""Shared training utilities."""

import copy
import gc
import json
import os
from abc import abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch._dynamo as dynamo
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

dynamo.config.suppress_errors = True
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(False)


RUN_PROFILE_EPOCH = False

TORCH_COMPILE_MODE = "reduce-overhead"


class Trainer:
    """Base trainer class with common training functionality."""

    def __init__(
        self,
        general_config: Any,
        model_config: Any,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """Initialize the Trainer with model, optimizer, and data loaders."""
        self.device = general_config.device
        self.start_time = datetime.now()
        self.model_config = model_config

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.num_validation_elements = len(self.validation_dataloader.dataset)  # type:ignore

        self.current_dropout_rate = self.model_config.dropout
        self.current_learning_rate = self.model_config.lr
        self.best_weights = None
        self.metric_minimum_loss = np.inf
        self.epoch_validation_loss = torch.zeros(general_config.num_species).to(
            self.device
        )
        self.stagnant_epochs = 0
        self.loss_per_epoch = []

    def save_loss_per_epoch(self) -> None:
        """Save the loss per epoch to a JSON file."""
        epochs_path = os.path.splitext(self.model_config.save_model_path)[0] + ".json"
        with open(epochs_path, "w") as f:
            json.dump(self.loss_per_epoch, f, indent=4)

    def print_final_time(self) -> None:
        """Print the total training time and number of epochs."""
        end_time = datetime.now()
        total_time = end_time - self.start_time
        print(f"Total Training Time: {total_time}")
        print(f"Total Epochs: {len(self.loss_per_epoch)}")

    def _save_checkpoint(self) -> None:
        """Save the model's state dictionary to a file."""
        checkpoint = self.model.state_dict()
        model_path = os.path.join(self.model_config.save_model_path)
        if self.model_config.save_model:
            torch.save(checkpoint, model_path)

    def set_dropout_rate(self, dropout_rate: float) -> None:
        """Set the dropout rate for all dropout layers in the model."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate
        self.current_dropout_rate = dropout_rate

    def _check_early_stopping(self) -> bool:
        """Check if training should stop due to stagnant epochs."""
        if self.stagnant_epochs >= self.model_config.stagnant_epoch_patience:
            print("Ending training early due to stagnant epochs.")
            return True
        return False

    def _check_minimum_loss(self) -> None:
        """Check if current validation loss is minimum and update best weights accordingly."""
        val_loss = self.epoch_validation_loss / self.num_validation_elements
        mean_loss = val_loss.mean().item()
        std_loss = val_loss.std().item()
        max_loss = val_loss.max().item()
        metric = mean_loss  # + std_loss + 0.5*max_loss

        if metric < self.metric_minimum_loss:
            print("**********************")
            print(
                f"New Minimum \nMean: {mean_loss:.3e} \nStd: {std_loss:.3e} \nMax: {max_loss:.3e} \nMetric: {metric:.3e} \nPercent Improvement: {(100 - metric * 100 / self.metric_minimum_loss):.3f}%"
            )
            self._save_checkpoint()
            self.best_weights = copy.deepcopy(self.model.state_dict())

            self.metric_minimum_loss = metric
            self.stagnant_epochs = 0
        else:
            self.stagnant_epochs += 1
            print(
                f"Stagnant {self.stagnant_epochs} \nMinimum: {self.metric_minimum_loss:.3e} \nMean: {mean_loss:.3e} \nStd: {std_loss:.3e} \nMax: {max_loss:.3e} \nMetric: {metric:.3e}"
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
                        f"Decreasing dropout rate to {self.current_dropout_rate:.4f} and settings lr to {self.current_learning_rate:.4f}."
                    )

            if (
                self.stagnant_epochs == self.model_config.lr_decay_patience + 1
                and self.best_weights
            ):
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
        """Run a single training and validation epoch."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _resolve_profile_batch_counts(
        self,
        warmup_batches: int,
        prof_batches: int,
    ) -> tuple[int, int]:
        """Clamp profile batch counts to the available training batches."""
        total_batches = len(self.training_dataloader)
        if total_batches <= 0:
            return 0, 0
        warmup_batches = min(warmup_batches, total_batches)
        prof_batches = min(prof_batches, total_batches - warmup_batches)
        if prof_batches == 0:
            prof_batches = 1
            warmup_batches = max(0, total_batches - prof_batches)
        return warmup_batches, prof_batches

    def _profile_trace_path(self, trace_name: str) -> str:
        """Build the profile trace path next to the model weights."""
        return os.path.join(
            os.path.dirname(self.model_config.save_model_path),
            f"{trace_name}_profile_trace.json",
        )

    def _profile_epoch(self) -> None:
        """Profile a short training run before the main training loop."""
        return

    def train(self) -> None:
        """Run training loop until early stopping criterion is met."""
        gc.collect()
        if str(self.device) == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        if RUN_PROFILE_EPOCH:
            print("Running profiling epoch.")
            self._profile_epoch()
            print("Profiling epoch completed.")
            print()

        for epoch in range(9999999):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break

        gc.collect()
        if str(self.device) == "cuda":
            torch.cuda.empty_cache()
        print(f"\nTraining Complete. Trial Results: {self.metric_minimum_loss}")
        self.print_final_time()
        self.save_loss_per_epoch()


def load_objects(
    model: nn.Module,
    config: Any,
) -> tuple[torch.optim.Optimizer, ReduceLROnPlateau]:
    """Create optimizer and learning rate scheduler for the model."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
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
