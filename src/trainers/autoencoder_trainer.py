"""Autoencoder trainer with full training logic."""

import copy
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from components.autoencoder import Autoencoder
from data_loading import AutoencoderDataset, tensor_to_dataloader
from data_processing import Processing
from loss import Loss

from .base_trainer import BaseTrainer


class AutoencoderTrainer(BaseTrainer):
    """Trainer specialized for autoencoder models with reconstruction loss."""

    def __init__(self, cfg: DictConfig, root: Path):
        """Initialize AutoencoderTrainer with component-based configuration."""
        super().__init__(cfg, root)

        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Load preprocessed data
        preprocessed_dir = cfg.get("paths", {}).get(
            "preprocessed", "outputs/preprocessed"
        )
        preprocess_dir = (
            root / f"{preprocessed_dir}/{cfg.dataset.name}/{cfg.preprocessing.name}"
        )
        train_path = preprocess_dir / "autoencoder_train_preprocessed.pt"
        val_path = preprocess_dir / "autoencoder_val_preprocessed.pt"

        print(f"Loading preprocessed data from {preprocess_dir}")
        training_tensor = torch.load(train_path)
        validation_tensor = torch.load(val_path)

        # Create datasets and dataloaders
        training_dataset = AutoencoderDataset(training_tensor)
        validation_dataset = AutoencoderDataset(validation_tensor)

        self.training_dataloader = tensor_to_dataloader(cfg.component, training_dataset)
        self.validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=cfg.component.batch_size,
            pin_memory=True,
            num_workers=10,
            shuffle=False,
        )
        self.num_validation_elements = len(validation_dataset)

        # Create model
        self.model = Autoencoder(
            input_dim=cfg.dataset.n_species,
            latent_dim=cfg.component.latent_dim,
            hidden_dims=list(cfg.component.hidden_dims),
            noise=cfg.component.noise,
            dropout=cfg.component.dropout,
        ).to(self.device)

        # Create optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.component.lr,
            betas=tuple(cfg.component.betas),
            weight_decay=cfg.component.weight_decay,
            fused=self.device == "cuda",
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=cfg.component.lr_decay,
            patience=cfg.component.lr_decay_patience,
        )

        # Create processing and loss functions
        self.processing = Processing(cfg.dataset, self.device)
        self.loss_fn = Loss(self.processing, cfg.dataset, cfg.component)

        # Training state
        self.gradient_clipping = cfg.component.gradient_clipping
        self.current_dropout_rate = cfg.component.dropout
        self.current_learning_rate = cfg.component.lr
        self.best_weights: dict[str, Any] | None = None
        self.epoch_validation_loss = torch.zeros(cfg.dataset.n_species).to(self.device)
        self.stagnant_epochs = 0
        self.start_time = datetime.now()

        # Count parameters
        self.param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {self.param_count}")

    def save_weights(self):
        """Save model weights to disk."""
        torch.save(self.model.state_dict(), self.output_dir / "weights.pth")

        # Save latents minmax for emulator training
        self._save_latents_minmax()

    def _save_latents_minmax(self):
        """Compute and save min/max values of encoded latents."""
        min_, max_ = float("inf"), float("-inf")

        self.model.eval()
        with torch.no_grad():
            for features in self.validation_dataloader:
                if isinstance(features, (list, tuple)):
                    features = features[0]
                features = features.to(self.device, non_blocking=True)
                encoded = self.model.encode(features).cpu()
                min_ = min(min_, encoded.min().item())
                max_ = max(max_, encoded.max().item())

        minmax_np = np.array([min_, max_], dtype=np.float32)
        print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")

        # Save to both output dir and configured path if different
        np.save(self.output_dir / "latents_minmax.npy", minmax_np)

        if "latents_minmax_path" in self.cfg.component:
            # Ensure directory exists
            path = Path(self.cfg.component.latents_minmax_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, minmax_np)

    def set_dropout_rate(self, dropout_rate: float):
        """Update dropout rate for all dropout layers in the model."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate
        self.current_dropout_rate = dropout_rate

    def train_epoch(self):
        """Execute one epoch of autoencoder training."""
        self.training_dataloader.sampler.set_epoch(self.current_epoch)  # type:ignore

        tic = datetime.now()
        self.model.train()

        for features in self.training_dataloader:
            if isinstance(features, (list, tuple)):
                features = features[0]
            features = features.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.loss_fn.training(outputs, features)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clipping
            )
            self.optimizer.step()

        toc = datetime.now()
        print(f"Training Time: {toc - tic}")

        return loss.item()

    def validate_epoch(self):
        """Execute one epoch of autoencoder validation."""
        tic = datetime.now()
        self.model.eval()

        with torch.no_grad():
            for features in self.validation_dataloader:
                if isinstance(features, (list, tuple)):
                    features = features[0]
                features = features.to(self.device, non_blocking=True)

                encoded = self.model.encode(features)
                outputs = self.model.decode(encoded)

                loss = self.loss_fn.validation(outputs, features)
                self.epoch_validation_loss += loss

        toc = datetime.now()
        print(f"Validation Time: {toc - tic}")

        # Compute metric
        val_loss = self.epoch_validation_loss / self.num_validation_elements
        mean_loss = val_loss.mean().item()
        std_loss = val_loss.std().item()
        max_loss = val_loss.max().item()
        metric = mean_loss

        print(
            f"Mean: {mean_loss:.3e} | Std: {std_loss:.3e} | Max: {max_loss:.3e} | Metric: {metric:.3e}"
        )

        # Update best weights and check for improvement
        if metric < self.best_loss:
            pct_improvement = 100 - metric * 100 / self.best_loss
            print("**********************")
            print(f"New Minimum! Percent Improvement: {pct_improvement:.3f}%")
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

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.current_learning_rate

                    print(
                        f"Decreasing dropout rate to {self.current_dropout_rate:.4f} "
                        f"and setting lr to {self.current_learning_rate:.4f}."
                    )

            if (
                self.stagnant_epochs == self.cfg.component.lr_decay_patience + 1
                and self.best_weights is not None
            ):
                print("Reverting to previous best weights")
                self.model.load_state_dict(self.best_weights)

        # Reset validation loss accumulator
        self.epoch_validation_loss.zero_()

        # Update scheduler
        self.scheduler.step(metric)
        print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.3e}")
        print(f"Current Dropout Rate: {self.current_dropout_rate:.4f}")

        return {
            "val_loss": metric,
            "mean": mean_loss,
            "std": std_loss,
            "max": max_loss,
            "dropout": self.current_dropout_rate,
            "learning_rate": self.current_learning_rate,
        }

    def should_stop(self):
        """Check if training should stop due to stagnant progress."""
        if self.stagnant_epochs >= self.cfg.component.stagnant_epoch_patience:
            print("Ending training early due to stagnant epochs.")
            return True
        return False

    def train(self):
        """Main training loop with early stopping."""
        # Cleanup memory
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Call parent train method
        super().train()

        # Final cleanup
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Print final time
        end_time = datetime.now()
        total_time = end_time - self.start_time
        print(f"\nTotal Training Time: {total_time}")
        print(f"Total Epochs: {len(self.metrics)}")
