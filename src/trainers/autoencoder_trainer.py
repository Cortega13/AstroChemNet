"""Trains an autoencoder component."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.components.autoencoder import Autoencoder
from src.data_loading import AutoencoderDataset, tensor_to_dataloader
from src.data_processing import Processing
from src.loss import Loss
from src.trainers.base_trainer import BaseTrainer


def _resolve_preprocess_dir(cfg: Any, root: Path) -> Path:
    """Resolve the preprocessing output directory."""
    return root / cfg.paths.preprocessed_dir / cfg.dataset.name / cfg.preprocessing.name


def _load_tensors(preprocess_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load preprocessed training and validation tensors."""
    train_path = preprocess_dir / "autoencoder_train_preprocessed.pt"
    val_path = preprocess_dir / "autoencoder_val_preprocessed.pt"
    return torch.load(train_path), torch.load(val_path)


def _build_dataloaders(
    cfg: Any,
    training_tensor: torch.Tensor,
    validation_tensor: torch.Tensor,
) -> tuple[DataLoader, DataLoader, int]:
    """Create training and validation dataloaders."""
    training_dataset = AutoencoderDataset(training_tensor)
    validation_dataset = AutoencoderDataset(validation_tensor)
    training_dataloader = tensor_to_dataloader(cfg.component, training_dataset)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.component.batch_size,
        pin_memory=True,
        num_workers=10,
        shuffle=False,
    )
    return training_dataloader, validation_dataloader, len(validation_dataset)


def _build_model(cfg: Any, device: str) -> Autoencoder:
    """Build the autoencoder model."""
    return Autoencoder(
        input_dim=cfg.dataset.n_species,
        latent_dim=cfg.component.latent_dim,
        hidden_dims=list(cfg.component.hidden_dims),
        noise=cfg.component.noise,
        dropout=cfg.component.dropout,
    ).to(device)


def _build_optimizer(cfg: Any, model: Autoencoder, device: str) -> optim.Optimizer:
    """Build the optimizer for autoencoder training."""
    return optim.AdamW(
        model.parameters(),
        lr=cfg.component.lr,
        betas=tuple(cfg.component.betas),
        weight_decay=cfg.component.weight_decay,
        fused=device == "cuda",
    )


def _build_scheduler(cfg: Any, optimizer: optim.Optimizer) -> ReduceLROnPlateau:
    """Build the learning rate scheduler."""
    return ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.component.lr_decay,
        patience=cfg.component.lr_decay_patience,
    )


def _build_processing(cfg: Any, device: str) -> Processing:
    """Build processing functions for scaling."""
    return Processing(cfg.dataset, device)


def _build_loss(processing: Processing, cfg: Any) -> Loss:
    """Build the loss function wrapper."""
    return Loss(processing, cfg.dataset, cfg.component)


def _count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def _extract_features(batch: object) -> torch.Tensor:
    """Extract feature tensor from a dataloader batch."""
    if isinstance(batch, (list, tuple)):
        return batch[0]
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError(f"Unexpected batch type: {type(batch)}")


class AutoencoderTrainer(BaseTrainer):
    """Trains an autoencoder with reconstruction loss."""

    def __init__(self, cfg: Any, root: Path) -> None:
        """Initialize AutoencoderTrainer."""
        super().__init__(cfg, root)
        preprocess_dir = _resolve_preprocess_dir(cfg, root)
        print(f"Loading preprocessed data from {preprocess_dir}")
        training_tensor, validation_tensor = _load_tensors(preprocess_dir)
        (
            self.training_dataloader,
            self.validation_dataloader,
            self.num_validation_elements,
        ) = _build_dataloaders(cfg, training_tensor, validation_tensor)
        model = _build_model(cfg, self.device)
        self.model = model
        self.optimizer = _build_optimizer(cfg, model, self.device)
        self.scheduler = _build_scheduler(cfg, self.optimizer)
        self.processing = _build_processing(cfg, self.device)
        self.loss_fn = _build_loss(self.processing, cfg)
        self.gradient_clipping = float(cfg.component.gradient_clipping)
        self.epoch_validation_loss = torch.zeros(cfg.dataset.n_species).to(self.device)
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
                    batch_features = _extract_features(batch).to(
                        self.device, non_blocking=True
                    )
                    encoded = model.encode(batch_features).cpu()
                    min_ = min(min_, encoded.min().item())
                    max_ = max(max_, encoded.max().item())
        minmax_np = np.array([min_, max_], dtype=np.float32)
        print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")
        np.save(self.output_dir / "latents_minmax.npy", minmax_np)
        if self.cfg.component.latents_minmax_path:
            path = Path(self.cfg.component.latents_minmax_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, minmax_np)

    def train_epoch(self) -> float:
        """Train the autoencoder for one epoch."""
        sampler = self.training_dataloader.sampler
        if hasattr(sampler, "set_epoch") and callable(sampler.set_epoch):
            sampler.set_epoch(self.current_epoch)
        tic = datetime.now()
        if self.model:
            self.model.train()
        loss = torch.tensor(0.0)
        model = cast(Autoencoder, self.model)
        for batch in self.training_dataloader:
            batch_features = _extract_features(batch).to(self.device, non_blocking=True)
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
                batch_features = _extract_features(batch).to(
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
