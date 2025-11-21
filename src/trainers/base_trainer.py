"""Base trainer module."""

import json
from pathlib import Path

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

        # Training state
        self.metrics = []
        self.best_loss = float("inf")
        self.current_epoch = 0
        self.param_count = 0

    def train(self):
        """Execute full training loop."""
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
            if not isinstance(val_metrics, dict):
                val_metrics = {"val_loss": val_metrics}

            val_loss = val_metrics.get("val_loss", float("inf"))

            # Track best
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_weights()

            # Log metrics
            epoch_metrics = {
                "epoch": epoch,
                "best_val_loss": self.best_loss,
            }
            epoch_metrics.update(train_metrics)
            epoch_metrics.update(val_metrics)

            self.metrics.append(epoch_metrics)

            # Early stopping check
            if self.should_stop():
                break

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

    def save_weights(self):
        """Save model weights."""
        raise NotImplementedError

    def train_epoch(self):
        """Train one epoch, return loss."""
        raise NotImplementedError

    def validate_epoch(self):
        """Validate, return loss."""
        raise NotImplementedError

    def should_stop(self):
        """Check early stopping criteria."""
        raise NotImplementedError
