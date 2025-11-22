"""Base trainer module."""

import copy
import gc
import json
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


class BaseTrainer:
    """Base trainer class."""

    def __init__(self, cfg: DictConfig, root: Path):
        """Initialize BaseTrainer."""
        self.cfg = cfg
        self.root = root
        weights_dir = cfg.get("paths", {}).get("weights", "outputs/weights")
        self.output_dir = root / f"{weights_dir}/{cfg.component.name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Training state
        self.metrics = []
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.param_count = 0
        self.stagnant_epochs = 0
        self.start_time: datetime | None = None
        self.best_weights: dict[str, Any] | None = None
        self.current_dropout_rate = cfg.component.dropout
        self.current_learning_rate = cfg.component.lr

        # Placeholders for model components (to be set by subclasses)
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None

    def train(self):
        """Execute full training loop."""
        self._setup_training()

        # Save config snapshot
        OmegaConf.save(self.cfg, self.output_dir / "config.yaml")

        # Training loop
        for epoch in range(self.cfg.component.epochs):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            # Handle if methods return just loss (float) or dict
            if not isinstance(train_metrics, dict):
                train_metrics = {"train_loss": train_metrics}

            # Process validation results (adaptive dropout, scheduler, best weights)
            # If validate_epoch returns a dict, we assume it's already processed or we extract loss
            if isinstance(val_metrics, dict):
                val_loss = val_metrics.get("val_loss", float("inf"))
                processed_metrics = val_metrics
            else:
                val_loss = val_metrics
                processed_metrics = self._process_validation_results(val_loss)

            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "best_val_loss": self.best_loss,
            }
            epoch_metrics.update(train_metrics)
            epoch_metrics.update(processed_metrics)

            self.metrics.append(epoch_metrics)

            # Early stopping check
            if self.should_stop():
                break

        self._teardown_training()

    def _setup_training(self):
        """Prepare for training (GC, timing)."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.start_time = datetime.now()

    def _teardown_training(self):
        """Cleanup after training."""
        # Save final metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save summary
        summary = {
            "best_val_loss": self.best_loss,
            "final_epoch": self.current_epoch,
            "total_params": self.param_count,
            "dataset": self.cfg.dataset.name,
            "preprocessing": self.cfg.preprocessing.name,
            "component_type": self.cfg.component.type,
        }
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Final cleanup
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Print final time
        if self.start_time:
            end_time = datetime.now()
            total_time = end_time - self.start_time
            print(f"\nTotal Training Time: {total_time}")
        print(f"Total Epochs: {len(self.metrics)}")

    def _process_validation_results(self, val_loss: float) -> dict[str, Any]:
        """Handle adaptive dropout, scheduler stepping, and best model tracking."""
        metric = val_loss

        # Update best weights and check for improvement
        if metric < self.best_loss:
            pct_improvement = 100 - metric * 100 / self.best_loss
            print("**********************")
            print(f"New Minimum! Percent Improvement: {pct_improvement:.3f}%")
            self.best_loss = metric
            self.save_weights()
            if self.model:
                self.best_weights = copy.deepcopy(self.model.state_dict())
            self.stagnant_epochs = 0
        else:
            self.stagnant_epochs += 1
            print(f"Stagnant {self.stagnant_epochs}")

            # Adaptive dropout reduction
            dropout_decay_patience = getattr(
                self.cfg.component, "dropout_decay_patience", 10
            )
            if self.stagnant_epochs % dropout_decay_patience == 0:
                reduction_factor = getattr(
                    self.cfg.component, "dropout_reduction_factor", 0.1
                )
                new_dropout = max(self.current_dropout_rate - reduction_factor, 0.0)
                if new_dropout != self.current_dropout_rate:
                    self.stagnant_epochs = 0
                    self.set_dropout_rate(new_dropout)

                    self.current_learning_rate = (
                        1e-3 if new_dropout <= 0.1 else self.cfg.component.lr
                    )

                    if self.optimizer:
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.current_learning_rate

                    print(
                        f"Decreasing dropout rate to {self.current_dropout_rate:.4f} "
                        f"and setting lr to {self.current_learning_rate:.4f}."
                    )

            if (
                self.stagnant_epochs == self.cfg.component.lr_decay_patience + 1
                and self.best_weights is not None
                and self.model
            ):
                print("Reverting to previous best weights")
                self.model.load_state_dict(self.best_weights)

        # Update scheduler
        if self.scheduler:
            self.scheduler.step(metric)
            if self.optimizer:
                print(
                    f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.3e}"
                )

        print(f"Current Dropout Rate: {self.current_dropout_rate:.4f}")

        return {
            "val_loss": metric,
            "dropout": self.current_dropout_rate,
            "learning_rate": self.current_learning_rate,
        }

    def set_dropout_rate(self, dropout_rate: float):
        """Update dropout rate for all dropout layers in the model."""
        if self.model:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = dropout_rate
        self.current_dropout_rate = dropout_rate

    def should_stop(self):
        """Check if training should stop due to stagnant progress."""
        if self.stagnant_epochs >= self.cfg.component.stagnant_epoch_patience:
            print("Ending training early due to stagnant epochs.")
            return True
        return False

    @abstractmethod
    def save_weights(self):
        """Save model weights."""
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self):
        """Train one epoch, return loss."""
        raise NotImplementedError

    @abstractmethod
    def validate_epoch(self):
        """Validate, return loss."""
        raise NotImplementedError
