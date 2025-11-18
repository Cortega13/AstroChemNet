"""Emulator trainer with full training logic for sequential autoregressive rollout."""

import copy
import gc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from components.autoencoder import Autoencoder
from components.emulator import Emulator
from data_loading import (
    EmulatorSequenceDataset,
    load_tensors_from_hdf5,
    tensor_to_dataloader,
)
from data_processing import Processing
from loss import Loss

from .base_trainer import BaseTrainer


class EmulatorTrainer(BaseTrainer):
    """Trainer specialized for sequential emulator models with autoregressive rollout."""

    def __init__(self, cfg: DictConfig, root: Path):
        """Initialize EmulatorTrainer with component-based configuration."""
        super().__init__(cfg, root)

        self.device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Load autoencoder config and weights
        ae_component_name = cfg.component.autoencoder_component
        ae_config_path = root / f"configs/components/{ae_component_name}.yaml"
        weights_dir = cfg.get("paths", {}).get("weights", "outputs/weights")
        ae_weights_path = root / f"{weights_dir}/{ae_component_name}/weights.pth"
        ae_latents_path = root / f"{weights_dir}/{ae_component_name}/latents_minmax.npy"

        if not ae_weights_path.exists():
            raise ValueError(
                f"Autoencoder weights not found: {ae_weights_path}\n"
                f"Train autoencoder first: python train.py component={ae_component_name}"
            )

        ae_cfg = OmegaConf.load(ae_config_path)

        # Load preprocessed sequential data
        preprocessed_dir = cfg.get("paths", {}).get(
            "preprocessed", "outputs/preprocessed"
        )
        preprocess_dir = (
            root / f"{preprocessed_dir}/{cfg.dataset.name}/{cfg.preprocessing.name}"
        )
        train_path = preprocess_dir / "training_seq.h5"
        val_path = preprocess_dir / "validation_seq.h5"

        print(f"Loading preprocessed sequential data from {preprocess_dir}")
        training_dataset_np, training_indices = load_tensors_from_hdf5(
            str(preprocess_dir), "training_seq"
        )
        validation_dataset_np, validation_indices = load_tensors_from_hdf5(
            str(preprocess_dir), "validation_seq"
        )

        # Create processing functions (needs autoencoder config for latent scaling)
        # Temporarily inject latents_minmax_path into ae_cfg
        ae_cfg.latents_minmax_path = str(ae_latents_path)
        self.processing = Processing(cfg.dataset, self.device, ae_cfg)

        # Create loss functions
        self.loss_fn = Loss(self.processing, cfg.dataset, cfg.component)

        # Create datasets
        training_dataset = EmulatorSequenceDataset(
            cfg.dataset, ae_cfg, training_dataset_np, training_indices
        )
        validation_dataset = EmulatorSequenceDataset(
            cfg.dataset, ae_cfg, validation_dataset_np, validation_indices
        )

        self.training_dataloader = tensor_to_dataloader(cfg.component, training_dataset)
        self.validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=cfg.component.batch_size,
            pin_memory=True,
            num_workers=10,
            shuffle=False,
        )
        self.num_validation_elements = len(validation_dataset)

        # Load pretrained autoencoder (frozen for inference)
        self.autoencoder = Autoencoder(
            input_dim=cfg.dataset.n_species,
            latent_dim=ae_cfg.latent_dim,
            hidden_dims=list(ae_cfg.hidden_dims),
            noise=ae_cfg.noise,
            dropout=ae_cfg.dropout,
        ).to(self.device)

        self.autoencoder.load_state_dict(
            torch.load(ae_weights_path, map_location="cpu")
        )
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        print(f"Loaded pretrained autoencoder from {ae_weights_path}")

        # Create emulator model
        # Input dim = physical params + latent dim
        emulator_input_dim = (
            cfg.dataset.physical_parameters.n_params + ae_cfg.latent_dim
        )
        emulator_output_dim = ae_cfg.latent_dim

        self.model = Emulator(
            input_dim=emulator_input_dim,
            output_dim=emulator_output_dim,
            hidden_dim=cfg.component.hidden_dims[0],  # Use first hidden dim
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

        # Training state
        self.latent_dim = ae_cfg.latent_dim
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

    def set_dropout_rate(self, dropout_rate: float):
        """Update dropout rate for all dropout layers in the model."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate
        self.current_dropout_rate = dropout_rate

    def train_epoch(self):
        """Execute one epoch of emulator training."""
        self.training_dataloader.sampler.set_epoch(self.current_epoch)

        tic = datetime.now()
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for phys, features, targets in self.training_dataloader:
            phys = phys.to(self.device, non_blocking=True)
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass through emulator
            outputs = self.model(phys, features)

            # Decode latents to abundance space for loss computation
            outputs = outputs.reshape(-1, self.latent_dim)
            outputs = self.processing.inverse_latent_components_scaling(outputs)
            outputs = self.autoencoder.decode(outputs)
            targets = targets.reshape(-1, self.cfg.dataset.n_species)

            loss = self.loss_fn.training(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clipping
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        toc = datetime.now()
        print(f"Training Time: {toc - tic}")

        return total_loss / max(num_batches, 1)

    def validate_epoch(self):
        """Execute one epoch of emulator validation."""
        tic = datetime.now()
        self.model.eval()

        with torch.no_grad():
            for phys, features, targets in self.validation_dataloader:
                phys = phys.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass through emulator
                outputs = self.model(phys, features)

                # Decode latents to abundance space
                outputs = outputs.reshape(-1, self.latent_dim)
                outputs = self.processing.inverse_latent_components_scaling(outputs)
                outputs = self.autoencoder.decode(outputs)
                outputs = outputs.reshape(targets.size(0), targets.size(1), -1)

                loss = self.loss_fn.validation(outputs, targets).mean(dim=0)
                self.epoch_validation_loss += loss.detach()

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

            if self.stagnant_epochs == self.cfg.component.lr_decay_patience + 1:
                if self.best_weights is not None:
                    print("Reverting to previous best weights")
                    self.model.load_state_dict(self.best_weights)

        # Update metrics for this epoch
        self.metrics[-1].update(
            {
                "mean": mean_loss,
                "std": std_loss,
                "max": max_loss,
                "dropout": self.current_dropout_rate,
                "learning_rate": self.current_learning_rate,
            }
        )

        # Reset validation loss accumulator
        self.epoch_validation_loss.zero_()

        # Update scheduler
        self.scheduler.step(metric)
        print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.3e}")
        print(f"Current Dropout Rate: {self.current_dropout_rate:.4f}")

        return metric

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
