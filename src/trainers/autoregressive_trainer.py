"""Autoregressive trainer with full training logic for sequential rollout."""

from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.components.autoencoder import Autoencoder
from src.components.emulator import Emulator
from src.data_loading import (
    AutoregressiveDataset,
    tensor_to_dataloader,
)
from src.data_processing import Processing
from src.loss import Loss
from src.trainers.base_trainer import BaseTrainer


class AutoregressiveTrainer(BaseTrainer):
    """Trainer specialized for autoregressive emulator models with sequential rollout."""

    def __init__(self, cfg: DictConfig, root: Path):
        """Initialize AutoregressiveTrainer with component-based configuration."""
        super().__init__(cfg, root)

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

        ae_cfg_raw = OmegaConf.load(ae_config_path)
        # Ensure ae_cfg is DictConfig (not ListConfig)
        if not isinstance(ae_cfg_raw, DictConfig):
            raise TypeError(f"Expected DictConfig, got {type(ae_cfg_raw)}")
        ae_cfg: DictConfig = ae_cfg_raw

        # Load preprocessed sequential data
        preprocessed_dir = cfg.get("paths", {}).get(
            "preprocessed", "outputs/preprocessed"
        )
        preprocess_dir = (
            root / f"{preprocessed_dir}/{cfg.dataset.name}/{cfg.preprocessing.name}"
        )
        print(f"Loading preprocessed sequential data from {preprocess_dir}")
        training_3d = torch.load(
            preprocess_dir / "autoregressive_train_preprocessed.pt"
        )
        validation_3d = torch.load(
            preprocess_dir / "autoregressive_val_preprocessed.pt"
        )

        # Create processing functions (needs autoencoder config for latent scaling)
        # Temporarily inject latents_minmax_path into ae_cfg
        ae_cfg.latents_minmax_path = str(ae_latents_path)
        self.processing = Processing(cfg.dataset, self.device, ae_cfg)

        # Create loss functions
        self.loss_fn = Loss(self.processing, cfg.dataset, cfg.component)

        # Create datasets with sliding window
        training_dataset = AutoregressiveDataset(
            cfg.dataset, ae_cfg, training_3d, cfg.component.horizon
        )
        validation_dataset = AutoregressiveDataset(
            cfg.dataset, ae_cfg, validation_3d, cfg.component.horizon
        )

        self.training_dataloader = tensor_to_dataloader(cfg.component, training_dataset)
        self.validation_dataloader = tensor_to_dataloader(
            cfg.component, validation_dataset, shuffle=False
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

        self.model: Emulator = Emulator(
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
        self.epoch_validation_loss = torch.zeros(cfg.dataset.n_species).to(self.device)

        # Count parameters
        self.param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {self.param_count}")

    def save_weights(self):
        """Save model weights to disk."""
        if self.model:
            torch.save(self.model.state_dict(), self.output_dir / "weights.pth")

    def train_epoch(self):
        """Execute one epoch of autoregressive training."""
        # Type guard for sampler with set_epoch method
        sampler = self.training_dataloader.sampler
        if hasattr(sampler, "set_epoch") and callable(
            getattr(sampler, "set_epoch", None)
        ):
            sampler.set_epoch(self.current_epoch)  # type: ignore[attr-defined]

        tic = datetime.now()
        if self.model:
            self.model.train()

        total_loss = 0.0
        num_batches = 0

        for phys, features, targets in self.training_dataloader:
            phys = phys.to(self.device, non_blocking=True)
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.optimizer and self.model:
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
        """Execute one epoch of autoregressive validation."""
        tic = datetime.now()
        if self.model:
            self.model.eval()

        with torch.no_grad():
            for phys, features, targets in self.validation_dataloader:
                phys = phys.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.model:
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

        # Reset validation loss accumulator
        self.epoch_validation_loss.zero_()

        return {
            "val_loss": metric,
            "mean": mean_loss,
            "std": std_loss,
            "max": max_loss,
        }
