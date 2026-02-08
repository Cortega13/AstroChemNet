"""Preprocesses datasets according to a selected method."""

import argparse
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.preprocessors import PREPROCESSOR_REGISTRY

ROOT = Path(__file__).parent.resolve()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument(
        "source",
        help="Dataset name (e.g., grav) or preprocessing method (e.g., initial)",
    )
    parser.add_argument(
        "method", help="Preprocessing method (e.g., initial, autoencoder)"
    )
    return parser.parse_args()


def _load_base_cfg() -> DictConfig | ListConfig:
    """Load the base Hydra configuration."""
    return OmegaConf.load(ROOT / "configs/config.yaml")


def _load_method_cfg(method: str) -> DictConfig | ListConfig:
    """Load the preprocessing method configuration."""
    return OmegaConf.load(ROOT / f"configs/preprocessing/{method}.yaml")


def _maybe_load_species(cfg: DictConfig | ListConfig) -> None:
    """Load species list from file if configured."""
    if hasattr(cfg, "species_file") and cfg.species_file:
        species = np.loadtxt(
            cfg.species_file, dtype=str, delimiter=" ", comments=None
        ).tolist()
        OmegaConf.update(cfg, "species", species, merge=False)


def _build_dataset_input_cfg(
    base_cfg: DictConfig | ListConfig, source: str, method_cfg: DictConfig | ListConfig
) -> DictConfig | ListConfig:
    """Build a merged config for dataset inputs."""
    dataset_cfg = OmegaConf.load(ROOT / f"configs/data/{source}.yaml")
    cfg = OmegaConf.merge(base_cfg, dataset_cfg, method_cfg)
    _maybe_load_species(cfg)
    return cfg


def _build_preprocessed_input_cfg(
    base_cfg: DictConfig | ListConfig, source: str, method_cfg: DictConfig | ListConfig
) -> tuple[DictConfig | ListConfig, str]:
    """Build a merged config for preprocessed inputs."""
    source_cfg = OmegaConf.load(ROOT / f"configs/preprocessing/{source}.yaml")
    dataset_name = OmegaConf.select(source_cfg, "input.source", default="grav")
    dataset_cfg = OmegaConf.load(ROOT / f"configs/data/{dataset_name}.yaml")
    cfg = OmegaConf.merge(base_cfg, dataset_cfg, source_cfg, method_cfg)
    _maybe_load_species(cfg)
    input_dir = str(ROOT / f"outputs/preprocessed/{dataset_name}/{source}")
    OmegaConf.update(cfg, "input_dir", input_dir, merge=False)
    return cfg, dataset_name


def _resolve_output_dir(
    input_type: str, source: str, method: str, dataset_name: str | None
) -> Path:
    """Resolve the output directory for preprocessing."""
    if input_type == "dataset":
        return ROOT / f"outputs/preprocessed/{source}/{method}"
    if dataset_name is None:
        raise ValueError("dataset_name is required for preprocessed inputs")
    return ROOT / f"outputs/preprocessed/{dataset_name}/{method}"


def main() -> None:
    """Entrypoint for preprocessing datasets."""
    args = _parse_args()
    base_cfg = _load_base_cfg()
    method_cfg = _load_method_cfg(args.method)
    input_type = OmegaConf.select(method_cfg, "input.type", default="dataset")
    dataset_name: str | None = None
    if input_type == "dataset":
        cfg = _build_dataset_input_cfg(base_cfg, args.source, method_cfg)
    else:
        cfg, dataset_name = _build_preprocessed_input_cfg(
            base_cfg, args.source, method_cfg
        )
    output_dir = _resolve_output_dir(input_type, args.source, args.method, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = PREPROCESSOR_REGISTRY[args.method](cfg)
    preprocessor.run(output_dir)
    print(f"✓ Preprocessed {args.source}/{args.method} → {output_dir}")


if __name__ == "__main__":
    main()
