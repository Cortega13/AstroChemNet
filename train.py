"""Training entrypoint."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from trainers import TRAINER_REGISTRY

ROOT = Path(__file__).parent.resolve()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entrypoint for training models."""
    # Component must be specified via CLI override
    if "component" not in cfg:
        raise ValueError(
            "Component not specified. Use: python train.py component=<component_name>"
        )

    # Load component config
    component_cfg = OmegaConf.load(ROOT / f"configs/components/{cfg.component}.yaml")

    # Load dataset config
    dataset_cfg = OmegaConf.load(ROOT / f"configs/data/{component_cfg.dataset}.yaml")

    # Load preprocessing config
    preprocessing_cfg = OmegaConf.load(
        ROOT / f"configs/preprocessing/{component_cfg.preprocessing_method}.yaml"
    )

    # Merge configs: global -> dataset -> preprocessing -> component
    merged_cfg = OmegaConf.merge(
        cfg,
        {
            "dataset": dataset_cfg,
            "preprocessing": preprocessing_cfg,
            "component": component_cfg,
        },
    )

    # Verify preprocessing exists
    preprocess_dir = (
        ROOT
        / f"{cfg.paths.preprocessed}/{dataset_cfg.name}/{component_cfg.preprocessing_method}"
    )
    if not preprocess_dir.exists():
        raise ValueError(
            f"Preprocessing not found: {preprocess_dir}\n"
            f"Run: python preprocess.py {dataset_cfg.name} {component_cfg.preprocessing_method}"
        )

    # Get trainer class and instantiate
    trainer_cls = TRAINER_REGISTRY[component_cfg.type]
    trainer = trainer_cls(merged_cfg, ROOT)

    # Train
    trainer.train()

    print(f"✓ Trained {component_cfg.name}")
    print(f"  Weights: {cfg.paths.weights}/{component_cfg.name}/")
    print(f"  Config:  {cfg.paths.weights}/{component_cfg.name}/config.yaml")
    print(f"  Metrics: {cfg.paths.weights}/{component_cfg.name}/metrics.json")


if __name__ == "__main__":
    main()
