"""Provides benchmark execution helpers; example: `from src.benchmark import run_benchmark`; assumptions: referenced component weights exist in `outputs/weights`."""

from pathlib import Path

from src.configs import (
    ComponentConfig,
    SurrogateConfig,
    load_component_config,
    load_runtime,
    load_surrogate_config,
)
from src.surrogates import SurrogateName

ROOT = Path(__file__).parent.parent.resolve()


def load_component_bundle(component_name: str, root: Path) -> tuple[ComponentConfig, Path]:
    """Loads component config and validates the weights file path."""
    cfg = load_component_config(root, component_name)
    runtime = load_runtime(root)
    weights_path = root / runtime.paths.weights_dir / component_name / "weights.pth"
    if not weights_path.exists():
        raise ValueError(f"Weights not found: {weights_path}")
    return cfg, weights_path


def _load_components(
    surrogate_name: str,
) -> tuple[SurrogateConfig, dict[str, dict[str, ComponentConfig | Path]]]:
    """Loads surrogate config and all referenced components."""
    surrogate_cfg = load_surrogate_config(ROOT, surrogate_name)
    components: dict[str, dict[str, ComponentConfig | Path]] = {}
    for role, component_name in surrogate_cfg.components.items():
        cfg, weights = load_component_bundle(component_name, ROOT)
        components[role] = {"config": cfg, "weights": weights}
    return surrogate_cfg, components


def run_benchmark(surrogate_name: str) -> None:
    """Executes benchmark run for a configured surrogate."""
    surrogate_cfg, components = _load_components(surrogate_name)
    surrogate_cls = SurrogateName.from_name(surrogate_cfg.name).to_class()
    surrogate = surrogate_cls(surrogate_cfg, components, ROOT)
    results = surrogate.benchmark()
    print(f"✓ Benchmark: {surrogate_name}")
    print(results)
