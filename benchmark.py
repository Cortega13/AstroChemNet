"""benchmark.py."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.surrogates import SURROGATE_REGISTRY

ROOT = Path(__file__).parent.resolve()


def load_component_config(component_name: str, root: Path):
    """Load component config and its weights path."""
    config_path = root / f"configs/components/{component_name}.yaml"
    weights_path = root / f"outputs/weights/{component_name}/weights.pth"

    if not weights_path.exists():
        raise ValueError(f"Weights not found: {weights_path}")

    return OmegaConf.load(config_path), weights_path


def main():
    """Entrypoint for benchmarking our models against validation."""
    parser = argparse.ArgumentParser(description="Benchmark surrogate model")
    parser.add_argument("surrogate", help="Surrogate config name")
    args = parser.parse_args()

    surrogate_cfg = OmegaConf.load(ROOT / f"configs/surrogates/{args.surrogate}.yaml")

    # Load all component configs and weights
    components = {}
    for role, component_name in surrogate_cfg.components.items():
        cfg, weights = load_component_config(component_name, ROOT)
        components[role] = {"config": cfg, "weights": weights}

    # Instantiate and benchmark
    surrogate = SURROGATE_REGISTRY[surrogate_cfg.name](surrogate_cfg, components, ROOT)
    results = surrogate.benchmark()

    print(f"✓ Benchmark: {args.surrogate}")
    print(results)


if __name__ == "__main__":
    main()
