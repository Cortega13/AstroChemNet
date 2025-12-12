"""Trains a configured component model."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.trainers import TRAINER_REGISTRY

ROOT = Path(__file__).parent.resolve()


def _load_component_cfg(cfg: DictConfig) -> DictConfig:
    """Load the component configuration."""
    return OmegaConf.load(ROOT / f"configs/components/{cfg.component}.yaml")


def _load_dataset_cfg(component_cfg: DictConfig) -> DictConfig:
    """Load the dataset configuration."""
    return OmegaConf.load(ROOT / f"configs/data/{component_cfg.dataset}.yaml")


def _load_preprocessing_cfg(component_cfg: DictConfig) -> DictConfig:
    """Load the preprocessing configuration."""
    return OmegaConf.load(
        ROOT / f"configs/preprocessing/{component_cfg.preprocessing_method}.yaml"
    )


def _merge_cfgs(
    cfg: DictConfig,
    dataset_cfg: DictConfig,
    preprocessing_cfg: DictConfig,
    component_cfg: DictConfig,
) -> DictConfig:
    """Merge global, dataset, preprocessing, and component configs."""
    return OmegaConf.merge(
        cfg,
        {
            "dataset": dataset_cfg,
            "preprocessing": preprocessing_cfg,
            "component": component_cfg,
        },
    )


def _verify_preprocessing_exists(
    cfg: DictConfig, dataset_cfg: DictConfig, component_cfg: DictConfig
) -> Path:
    """Verify that preprocessing output exists."""
    preprocess_dir = (
        ROOT
        / f"{cfg.paths.preprocessed}/{dataset_cfg.name}/{component_cfg.preprocessing_method}"
    )
    if not preprocess_dir.exists():
        raise ValueError(
            f"Preprocessing not found: {preprocess_dir}\n"
            f"Run: python preprocess.py {dataset_cfg.name} {component_cfg.preprocessing_method}"
        )
    return preprocess_dir


def run_training(cfg: DictConfig) -> None:
    """Run training for the selected component."""
    if "component" not in cfg:
        raise ValueError(
            "Component not specified. Use: python train.py component=<component_name>"
        )
    component_cfg = _load_component_cfg(cfg)
    dataset_cfg = _load_dataset_cfg(component_cfg)
    preprocessing_cfg = _load_preprocessing_cfg(component_cfg)
    merged_cfg = _merge_cfgs(cfg, dataset_cfg, preprocessing_cfg, component_cfg)
    _verify_preprocessing_exists(cfg, dataset_cfg, component_cfg)
    trainer_cls = TRAINER_REGISTRY[component_cfg.type]
    trainer = trainer_cls(merged_cfg, ROOT)
    trainer.train()
    print(f"✓ Trained {component_cfg.name}")
    print(f"  Weights: {cfg.paths.weights}/{component_cfg.name}/")
    print(f"  Config:  {cfg.paths.weights}/{component_cfg.name}/config.yaml")
    print(f"  Metrics: {cfg.paths.weights}/{component_cfg.name}/metrics.json")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entrypoint for Hydra-based training."""
    run_training(cfg)


if __name__ == "__main__":
    main()
