"""preprocess.py."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from preprocessors import PREPROCESSOR_REGISTRY

ROOT = Path(__file__).parent.resolve()


def main():
    """Entrypoint for preprocessing datasets."""
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("dataset", help="Dataset name (e.g., grav)")
    parser.add_argument("method", help="Preprocessing method (e.g., sequential)")
    args = parser.parse_args()

    # Load base config
    base_cfg = OmegaConf.load(ROOT / "configs/config.yaml")

    # Load dataset and method configs
    dataset_cfg = OmegaConf.load(ROOT / f"configs/data/{args.dataset}.yaml")
    method_cfg = OmegaConf.load(ROOT / f"configs/preprocessing/{args.method}.yaml")

    # Merge configs: base <- dataset <- method
    # This ensures dataset/method settings override base defaults
    cfg = OmegaConf.merge(base_cfg, dataset_cfg, method_cfg)

    preprocessor = PREPROCESSOR_REGISTRY[args.method](cfg)
    output_dir = ROOT / f"outputs/preprocessed/{args.dataset}/{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor.run(output_dir)
    print(f"✓ Preprocessed {args.dataset}/{args.method} → {output_dir}")


if __name__ == "__main__":
    main()
