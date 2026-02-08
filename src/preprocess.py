"""Provides preprocessing execution helpers; example: `from src.preprocess import run_preprocess`; assumptions: source data files are available for the selected dataset/method."""

from pathlib import Path

from src.configs import build_preprocess_run_config
from src.preprocessors import PREPROCESSOR_REGISTRY

ROOT = Path(__file__).parent.parent.resolve()


def _resolve_output_dir(root: Path, preprocessed_dir: str, dataset_name: str, method: str) -> Path:
    """Resolves the preprocessing output directory."""
    return root / preprocessed_dir / dataset_name / method


def run_preprocess(source: str, method: str) -> None:
    """Executes preprocessing for the selected source and method."""
    cfg, dataset_name = build_preprocess_run_config(ROOT, source, method)
    output_dir = _resolve_output_dir(ROOT, cfg.paths.preprocessed_dir, dataset_name, method)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = PREPROCESSOR_REGISTRY[method](cfg)
    preprocessor.run(output_dir)
    print(f"✓ Preprocessed {source}/{method} → {output_dir}")
