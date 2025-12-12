"""Benchmarks a surrogate against validation data."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from src.surrogates import SURROGATE_REGISTRY

ROOT = Path(__file__).parent.resolve()


def load_component_config(component_name: str, root: Path) -> tuple[OmegaConf, Path]:
    """Load a component config and weights path."""
    config_path = root / f"configs/components/{component_name}.yaml"
    weights_path = root / f"outputs/weights/{component_name}/weights.pth"

    if not weights_path.exists():
        raise ValueError(f"Weights not found: {weights_path}")

    return OmegaConf.load(config_path), weights_path


def _parse_args() -> argparse.Namespace:
    """Parse CLI args for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark surrogate model")
    parser.add_argument("surrogate", help="Surrogate config name")
    return parser.parse_args()


def _load_surrogate_cfg(name: str) -> OmegaConf:
    """Load a surrogate configuration by name."""
    return OmegaConf.load(ROOT / f"configs/surrogates/{name}.yaml")


def _load_components(surrogate_cfg: OmegaConf) -> dict[str, dict[str, object]]:
    """Load component configs and weight paths for a surrogate."""
    components: dict[str, dict[str, object]] = {}
    for role, component_name in surrogate_cfg.components.items():
        cfg, weights = load_component_config(component_name, ROOT)
        components[role] = {"config": cfg, "weights": weights}
    return components


def main() -> None:
    """Entrypoint for benchmarking surrogates."""
    args = _parse_args()
    surrogate_cfg = _load_surrogate_cfg(args.surrogate)
    components = _load_components(surrogate_cfg)
    surrogate = SURROGATE_REGISTRY[surrogate_cfg.name](surrogate_cfg, components, ROOT)
    results = surrogate.benchmark()

    print(f"✓ Benchmark: {args.surrogate}")
    print(results)


if __name__ == "__main__":
    main()
