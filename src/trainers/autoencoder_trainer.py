"""Trains an autoencoder component."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.components.autoencoder import Autoencoder
from src.configs import AutoencoderConfig, TrainingRunConfig
from src.data_loading import AutoencoderDataset, tensor_to_dataloader
from src.data_processing import Processing
from src.loss import Loss
from src.trainers.base_trainer import BaseTrainer


def _resolve_preprocess_dir(run_config: TrainingRunConfig, root: Path) -> Path:
    """Resolve the preprocessing output directory."""
    return (
        root
        / run_config.paths.preprocessed_dir
        / run_config.dataset.name
        / run_config.preprocessing.name
    )


def _load_tensors(preprocess_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load preprocessed training and validation tensors."""
    train_path = preprocess_dir / "autoencoder_train_preprocessed.pt"
    val_path = preprocess_dir / "autoencoder_val_preprocessed.pt"
    return torch.load(train_path), torch.load(val_path)


def _build_dataloaders(
    component_config: AutoencoderConfig,
    training_tensor: torch.Tensor,
    validation_tensor: torch.Tensor,
    device: str,
) -> tuple[DataLoader, DataLoader, int]:
    """Create training and validation dataloaders."""
    training_dataset = AutoencoderDataset(training_tensor)
    validation_dataset = AutoencoderDataset(validation_tensor)
    training_dataloader = tensor_to_dataloader(
        component_config, training_dataset, device
    )
    validation_dataloader = tensor_to_dataloader(
        component_config, validation_dataset, device, shuffle=False
    )
    return training_dataloader, validation_dataloader, len(validation_dataset)


def _build_model(
    run_config: TrainingRunConfig,
    component_config: AutoencoderConfig,
    device: str,
) -> Autoencoder:
    """Build the autoencoder model."""
    return Autoencoder(
        input_dim=run_config.dataset.num_species,
        latent_dim=component_config.latent_dim,
        hidden_dims=list(component_config.hidden_dims),
        noise=component_config.noise,
        dropout=component_config.dropout,
    ).to(device)


def _build_optimizer_scheduler(
    component_config: AutoencoderConfig,
    model: Autoencoder,
    device: str,
) -> tuple[optim.Optimizer, ReduceLROnPlateau]:
    """Build optimizer and scheduler for autoencoder training."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=component_config.lr,
        betas=component_config.betas,
        weight_decay=component_config.weight_decay,
        fused=device == "cuda",
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=component_config.lr_decay,
        patience=component_config.lr_decay_patience,
    )
    return optimizer, scheduler


def _build_loss(
    processing: Processing,
    run_config: TrainingRunConfig,
    component_config: AutoencoderConfig,
) -> Loss:
    """Build the loss function wrapper."""
    return Loss(processing, run_config.dataset, component_config)


def _count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


class AutoencoderTrainer(BaseTrainer):
    """Trains an autoencoder with reconstruction loss."""

    def __init__(self, run_config: TrainingRunConfig, root: Path) -> None:
        """Initialize AutoencoderTrainer."""
        super().__init__(run_config, root)
        self.component_config = cast(AutoencoderConfig, run_config.component)
        preprocess_dir = _resolve_preprocess_dir(run_config, root)
        print(f"Loading preprocessed data from {preprocess_dir}")
        training_tensor, validation_tensor = _load_tensors(preprocess_dir)
        (
            self.training_dataloader,
            self.validation_dataloader,
            self.num_validation_elements,
        ) = _build_dataloaders(
            self.component_config,
            training_tensor,
            validation_tensor,
            self.device,
        )
        model = _build_model(run_config, self.component_config, self.device)
        self.model = model
        self.optimizer, self.scheduler = _build_optimizer_scheduler(
            self.component_config,
            model,
            self.device,
        )
        self.processing = Processing(run_config.dataset, self.device)
        self.loss_fn = _build_loss(self.processing, run_config, self.component_config)
        self.gradient_clipping = float(self.component_config.gradient_clipping)
        self.epoch_validation_loss = torch.zeros(run_config.dataset.num_species).to(
            self.device
        )
        self.param_count = _count_parameters(self.model)
        print(f"Total Parameters: {self.param_count}")

    def save_weights(self) -> None:
        """Save model weights to disk."""
        if self.model:
            torch.save(self.model.state_dict(), self.output_dir / "weights.pth")
        self._save_latents_minmax()

    def _save_latents_minmax(self) -> None:
        """Compute and save min/max values of encoded latents."""
        min_, max_ = float("inf"), float("-inf")
        if self.model:
            model = cast(Autoencoder, self.model)
            self.model.eval()
            with torch.no_grad():
                for batch in self.validation_dataloader:
                    batch_features = (
                        batch[0] if isinstance(batch, (list, tuple)) else batch
                    )
                    batch_features = cast(torch.Tensor, batch_features).to(
                        self.device, non_blocking=True
                    )
                    encoded = model.encode(batch_features).cpu()
                    min_ = min(min_, encoded.min().item())
                    max_ = max(max_, encoded.max().item())
        minmax_np = np.array([min_, max_], dtype=np.float32)
        print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")
        np.save(self.output_dir / "latents_minmax.npy", minmax_np)
        if self.component_config.latents_minmax_path:
            path = Path(self.component_config.latents_minmax_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, minmax_np)

    def train_epoch(self) -> float:
        """Train the autoencoder for one epoch."""
        self.training_dataloader.sampler.set_epoch(self.current_epoch)  # type:ignore
        tic = datetime.now()
        if self.model:
            self.model.train()
        loss = torch.tensor(0.0)
        model = cast(Autoencoder, self.model)
        for batch in self.training_dataloader:
            batch_features = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_features = cast(torch.Tensor, batch_features).to(
                self.device, non_blocking=True
            )
            if self.optimizer and self.model:
                self.optimizer.zero_grad()
                outputs = model(batch_features)
                loss = self.loss_fn.training(outputs, batch_features)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.gradient_clipping
                )
                self.optimizer.step()
        print(f"Training Time: {datetime.now() - tic}")
        return float(loss.item())

    def validate_epoch(self) -> dict[str, float]:
        """Validate the autoencoder for one epoch."""
        tic = datetime.now()
        if self.model:
            self.model.eval()

        model = cast(Autoencoder, self.model)
        with torch.no_grad():
            for batch in self.validation_dataloader:
                batch_features = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_features = cast(torch.Tensor, batch_features).to(
                    self.device, non_blocking=True
                )
                if self.model:
                    encoded = model.encode(batch_features)
                    outputs = model.decode(encoded)
                    loss = self.loss_fn.validation(outputs, batch_features)
                    self.epoch_validation_loss += loss
        print(f"Validation Time: {datetime.now() - tic}")
        val_loss = self.epoch_validation_loss / self.num_validation_elements
        mean_loss = float(val_loss.mean().item())
        std_loss = float(val_loss.std().item())
        max_loss = float(val_loss.max().item())
        metric = mean_loss
        print(
            f"Mean: {mean_loss:.3e} | Std: {std_loss:.3e} | "
            f"Max: {max_loss:.3e} | Metric: {metric:.3e}"
        )
        self.epoch_validation_loss.zero_()
        return {"val_loss": metric, "mean": mean_loss, "std": std_loss, "max": max_loss}
