"""Trains a latent-space emulator component."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.components.autoencoder import Autoencoder
from src.components.emulator import Emulator
from src.configs import AutoencoderConfig, ComponentName
from src.data_loading import AutoregressiveDataset, tensor_to_dataloader
from src.data_processing import Processing
from src.loss import Loss
from src.trainers.base_trainer import BaseTrainer


def _resolve_ae_paths(cfg: Any, root: Path) -> tuple[str, Path, Path]:
    """Resolve autoencoder config, weights, and latents paths."""
    ae_component_name = cfg.component.autoencoder_component
    ae_weights_path = root / cfg.paths.weights_dir / ae_component_name / "weights.pth"
    ae_latents_path = (
        root / cfg.paths.weights_dir / ae_component_name / "latents_minmax.npy"
    )
    return ae_component_name, ae_weights_path, ae_latents_path


def _load_ae_cfg(root: Path, component_name: str) -> AutoencoderConfig:
    """Load autoencoder component configuration."""
    ae_cfg = ComponentName(component_name).config()
    if not isinstance(ae_cfg, AutoencoderConfig):
        raise TypeError(f"Expected autoencoder component, got {type(ae_cfg).__name__}")
    return ae_cfg


def _resolve_preprocess_dir(cfg: Any, root: Path) -> Path:
    """Resolve preprocessing output directory."""
    return root / cfg.paths.preprocessed_dir / cfg.dataset.name / cfg.preprocessing.name


def _load_sequential_tensors(preprocess_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load autoregressive training and validation tensors."""
    train_path = preprocess_dir / "autoregressive_train_preprocessed.pt"
    val_path = preprocess_dir / "autoregressive_val_preprocessed.pt"
    return torch.load(train_path), torch.load(val_path)


def _build_processing(
    cfg: Any, device: str, ae_cfg: AutoencoderConfig, ae_latents_path: Path
) -> Processing:
    """Build processing functions including latent scaling."""
    ae_cfg.latents_minmax_path = str(ae_latents_path)
    return Processing(cfg.dataset, device, ae_cfg)


def _build_loss(processing: Processing, cfg: Any) -> Loss:
    """Build loss for emulator training."""
    return Loss(processing, cfg.dataset, cfg.component)


def _build_dataloaders(
    cfg: Any,
    ae_cfg: AutoencoderConfig,
    training_3d: torch.Tensor,
    validation_3d: torch.Tensor,
) -> tuple[DataLoader, DataLoader, int]:
    """Create training and validation dataloaders."""
    training_dataset = AutoregressiveDataset(
        cfg.dataset, ae_cfg, training_3d, cfg.component.horizon
    )
    validation_dataset = AutoregressiveDataset(
        cfg.dataset, ae_cfg, validation_3d, cfg.component.horizon
    )
    training_dataloader = tensor_to_dataloader(cfg.component, training_dataset)
    validation_dataloader = tensor_to_dataloader(
        cfg.component, validation_dataset, shuffle=False
    )
    return training_dataloader, validation_dataloader, len(validation_dataset)


def _load_frozen_autoencoder(
    cfg: Any, ae_cfg: AutoencoderConfig, ae_weights_path: Path, device: str
) -> Autoencoder:
    """Load a pretrained autoencoder in eval mode."""
    autoencoder = Autoencoder(
        input_dim=cfg.dataset.n_species,
        latent_dim=ae_cfg.latent_dim,
        hidden_dims=list(ae_cfg.hidden_dims),
        noise=ae_cfg.noise,
        dropout=ae_cfg.dropout,
    ).to(device)
    autoencoder.load_state_dict(torch.load(ae_weights_path, map_location="cpu"))
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    print(f"Loaded pretrained autoencoder from {ae_weights_path}")
    return autoencoder


def _build_emulator_model(cfg: Any, ae_cfg: AutoencoderConfig, device: str) -> Emulator:
    """Build the emulator model."""
    emulator_input_dim = cfg.dataset.n_params + ae_cfg.latent_dim
    emulator_output_dim = ae_cfg.latent_dim
    return Emulator(
        input_dim=emulator_input_dim,
        output_dim=emulator_output_dim,
        hidden_dim=int(cfg.component.hidden_dim),
        dropout=float(cfg.component.dropout),
    ).to(device)


def _build_optimizer(cfg: Any, model: Emulator, device: str) -> optim.Optimizer:
    """Build the optimizer for emulator training."""
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


def _extract_batch(batch: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract phys, features, and targets from a batch."""
    if not isinstance(batch, (list, tuple)) or len(batch) != 3:
        raise TypeError(f"Unexpected batch type: {type(batch)}")
    phys, features, targets = batch
    if not all(isinstance(x, torch.Tensor) for x in (phys, features, targets)):
        raise TypeError("Batch elements must be tensors")
    return phys, features, targets


class EmulatorTrainer(BaseTrainer):
    """Trains a latent-space autoregressive emulator."""

    def __init__(self, cfg: Any, root: Path) -> None:
        """Initialize EmulatorTrainer."""
        super().__init__(cfg, root)
        ae_component_name, ae_weights_path, ae_latents_path = _resolve_ae_paths(
            cfg, root
        )
        if not ae_weights_path.exists():
            raise ValueError(
                f"Autoencoder weights not found: {ae_weights_path}\n"
                f"Train autoencoder first: python run.py train {cfg.component.autoencoder_component}"
            )
        ae_cfg = _load_ae_cfg(root, ae_component_name)
        preprocess_dir = _resolve_preprocess_dir(cfg, root)
        print(f"Loading preprocessed sequential data from {preprocess_dir}")
        training_3d, validation_3d = _load_sequential_tensors(preprocess_dir)
        self.processing = _build_processing(cfg, self.device, ae_cfg, ae_latents_path)
        self.loss_fn = _build_loss(self.processing, cfg)
        (
            self.training_dataloader,
            self.validation_dataloader,
            self.num_validation_elements,
        ) = _build_dataloaders(cfg, ae_cfg, training_3d, validation_3d)
        self.autoencoder = _load_frozen_autoencoder(
            cfg, ae_cfg, ae_weights_path, self.device
        )
        self.model = _build_emulator_model(cfg, ae_cfg, self.device)
        self.optimizer = _build_optimizer(cfg, self.model, self.device)
        self.scheduler = _build_scheduler(cfg, self.optimizer)
        self.latent_dim = int(ae_cfg.latent_dim)
        self.gradient_clipping = float(cfg.component.gradient_clipping)
        self.epoch_validation_loss = torch.zeros(cfg.dataset.n_species).to(self.device)
        self.param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {self.param_count}")

    def save_weights(self) -> None:
        """Save model weights to disk."""
        if self.model:
            torch.save(self.model.state_dict(), self.output_dir / "weights.pth")

    def _decode_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Decode emulator latent outputs to abundance space."""
        flat = outputs.reshape(-1, self.latent_dim)
        flat = self.processing.inverse_latents_scaling(flat)
        decoded = self.autoencoder.decode(flat)
        return decoded

    def train_epoch(self) -> float:
        """Train the emulator for one epoch."""
        self.training_dataloader.sampler.set_epoch(self.current_epoch)  # type:ignore
        tic = datetime.now()
        if self.model:
            self.model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in self.training_dataloader:
            phys, features, targets = _extract_batch(batch)
            phys = phys.to(self.device, non_blocking=True)
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if self.optimizer and self.model:
                self.optimizer.zero_grad()
                outputs = self.model(phys, features)
                decoded = self._decode_outputs(outputs)
                targets_flat = targets.reshape(-1, self.cfg.dataset.n_species)
                loss = self.loss_fn.training(decoded, targets_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
                self.optimizer.step()
                total_loss += float(loss.item())
                num_batches += 1
        print(f"Training Time: {datetime.now() - tic}")
        return total_loss / max(num_batches, 1)

    def validate_epoch(self) -> dict[str, float]:
        """Validate the emulator for one epoch."""
        tic = datetime.now()
        if self.model:
            self.model.eval()
        with torch.no_grad():
            for batch in self.validation_dataloader:
                phys, features, targets = _extract_batch(batch)
                phys = phys.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.model:
                    outputs = self.model(phys, features)
                    decoded = self._decode_outputs(outputs).reshape(
                        targets.size(0), targets.size(1), -1
                    )
                    loss = self.loss_fn.validation(decoded, targets).mean(dim=0)
                    self.epoch_validation_loss += loss.detach()
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
