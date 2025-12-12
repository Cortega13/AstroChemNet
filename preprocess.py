"""preprocess.py."""

import argparse
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.preprocessors import PREPROCESSOR_REGISTRY

ROOT = Path(__file__).parent.resolve()


def main():
    """Entrypoint for preprocessing datasets."""
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument(
        "source",
        help="Dataset name (e.g., grav) or preprocessing method (e.g., initial)",
    )
    parser.add_argument(
        "method", help="Preprocessing method (e.g., initial, autoencoder)"
    )
    args = parser.parse_args()

    # Load base config
    base_cfg = OmegaConf.load(ROOT / "configs/config.yaml")

    # Load method config to determine input type
    method_cfg = OmegaConf.load(ROOT / f"configs/preprocessing/{args.method}.yaml")

    # Check input type from method config (use OmegaConf style access)
    input_type = OmegaConf.select(method_cfg, "input.type", default="dataset")

    if input_type == "dataset":
        # Load dataset config
        dataset_cfg = OmegaConf.load(ROOT / f"configs/data/{args.source}.yaml")
        cfg = OmegaConf.merge(base_cfg, dataset_cfg, method_cfg)

        # Load species list from file if species_file is specified
        if hasattr(cfg, "species_file") and cfg.species_file:
            species = np.loadtxt(
                cfg.species_file, dtype=str, delimiter=" ", comments=None
            ).tolist()
            OmegaConf.update(cfg, "species", species, merge=False)

        output_dir = ROOT / f"outputs/preprocessed/{args.source}/{args.method}"
    else:
        # Load source preprocessing config
        source_cfg = OmegaConf.load(ROOT / f"configs/preprocessing/{args.source}.yaml")
        # Get dataset from source config (via defaults or input.source)
        dataset_name = OmegaConf.select(source_cfg, "input.source", default="grav")
        dataset_cfg = OmegaConf.load(ROOT / f"configs/data/{dataset_name}.yaml")

        # Merge configs: base <- dataset <- source <- method
        cfg = OmegaConf.merge(base_cfg, dataset_cfg, source_cfg, method_cfg)

        # Load species list from file if species_file is specified
        if hasattr(cfg, "species_file") and cfg.species_file:
            species = np.loadtxt(
                cfg.species_file, dtype=str, delimiter=" ", comments=None
            ).tolist()
            OmegaConf.update(cfg, "species", species, merge=False)

        # Set input_dir to point to source preprocessing output
        input_dir = str(ROOT / f"outputs/preprocessed/{dataset_name}/{args.source}")
        OmegaConf.update(cfg, "input_dir", input_dir, merge=False)

        output_dir = ROOT / f"outputs/preprocessed/{dataset_name}/{args.method}"

    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = PREPROCESSOR_REGISTRY[args.method](cfg)
    preprocessor.run(output_dir)
    print(f"✓ Preprocessed {args.source}/{args.method} → {output_dir}")


if __name__ == "__main__":
    main()
