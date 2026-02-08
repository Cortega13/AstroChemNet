"""Provides training execution helpers; example: `from src.train import run_training`; assumptions: preprocessing outputs exist for the selected component."""

from pathlib import Path

from src.configs import build_training_config
from src.trainers import TRAINER_REGISTRY

ROOT = Path(__file__).parent.parent.resolve()


def _verify_preprocessing_exists(root: Path, component_name: str) -> None:
    """Verifies the preprocessing output directory exists before training."""
    cfg = build_training_config(root, component_name)
    preprocess_dir = (
        root / cfg.paths.preprocessed_dir / cfg.dataset.name / cfg.preprocessing.name
    )
    if not preprocess_dir.exists():
        raise ValueError(
            f"Preprocessing not found: {preprocess_dir}\n"
            f"Run: python run.py preprocess {cfg.dataset.name} {cfg.preprocessing.name}"
        )


def run_training(component_name: str) -> None:
    """Runs training for the selected component."""
    _verify_preprocessing_exists(ROOT, component_name)
    cfg = build_training_config(ROOT, component_name)
    trainer_cls = TRAINER_REGISTRY[cfg.component.trainingtype]
    trainer = trainer_cls(cfg, ROOT)
    trainer.train()
    print(f"✓ Trained {cfg.component.name}")
    print(f"  Weights: {cfg.paths.weights_dir}/{cfg.component.name}/")
    print(f"  Config:  {cfg.paths.weights_dir}/{cfg.component.name}/config.json")
    print(f"  Metrics: {cfg.paths.weights_dir}/{cfg.component.name}/metrics.json")
