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

    dataset_cfg = OmegaConf.load(ROOT / f"configs/datasets/{args.dataset}.yaml")
    method_cfg = OmegaConf.load(ROOT / f"configs/preprocessing/{args.method}.yaml")

    preprocessor = PREPROCESSOR_REGISTRY[args.method](dataset_cfg, method_cfg, ROOT)
    output_dir = ROOT / f"outputs/preprocessed/{args.dataset}/{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor.run(output_dir)
    print(f"✓ Preprocessed {args.dataset}/{args.method} → {output_dir}")


if __name__ == "__main__":
    main()
