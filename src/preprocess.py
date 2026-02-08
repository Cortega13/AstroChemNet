"""Provides preprocessing execution helpers; example: `from src.preprocess import run_preprocess`; assumptions: config names map to valid dataset/component preprocessing stages."""

from pathlib import Path

from src.configs import (
    ComponentName,
    DatasetName,
    PreprocessRunConfig,
    build_preprocess_run_config,
)
from src.preprocessors import PREPROCESSOR_REGISTRY

ROOT = Path(__file__).parent.parent.resolve()


def _resolve_shorthand_target(target: str) -> tuple[str, str]:
    """Resolves one-argument preprocess shorthand to source and method."""
    try:
        DatasetName(target)
        return target, "uclchem_grav"
    except ValueError:
        pass
    try:
        component = ComponentName(target).config()
    except ValueError as error:
        raise ValueError(
            f"Unknown preprocess target: {target}\n"
            "Use a dataset or component config name."
        ) from error
    return "uclchem_grav", component.preprocessing_method


def _resolve_output_dir(
    root: Path, preprocessed_dir: str, dataset_name: str, method: str
) -> Path:
    """Resolves the preprocessing output directory."""
    return root / preprocessed_dir / dataset_name / method


def _verify_uclchem_grav_preprocessing(
    cfg: PreprocessRunConfig, root: Path, dataset_name: str
) -> None:
    """Verifies uclchem_grav preprocessing exists for derived preprocessing methods."""
    if cfg.method.name == "uclchem_grav":
        return
    uclchem_grav_dir = _resolve_output_dir(
        root, cfg.paths.preprocessed_dir, dataset_name, "uclchem_grav"
    )
    if uclchem_grav_dir.exists():
        return
    raise ValueError(
        f"UCLCHEM grav preprocessing not found: {uclchem_grav_dir}\n"
        f"Run: python run.py preprocess {dataset_name}"
    )


def _verify_autoencoder_weights(cfg: PreprocessRunConfig, root: Path) -> None:
    """Verifies pretrained autoencoder weights exist for autoregressive preprocessing."""
    if cfg.method.name != "autoregressive" or cfg.autoencoder is None:
        return
    weights_rel = cfg.autoencoder.pretrained_model_path
    if weights_rel is None:
        raise ValueError("autoencoder.pretrained_model_path is required")
    weights_path = root / weights_rel
    if weights_path.exists():
        return
    raise ValueError(
        f"Pretrained autoencoder weights not found: {weights_path}\n"
        f"Run: python run.py train {cfg.autoencoder.name}"
    )


def run_preprocess(target: str) -> None:
    """Executes preprocessing for one dataset/component shorthand target."""
    source, method = _resolve_shorthand_target(target)
    cfg, dataset_name = build_preprocess_run_config(ROOT, source, method)
    _verify_uclchem_grav_preprocessing(cfg, ROOT, dataset_name)
    _verify_autoencoder_weights(cfg, ROOT)
    output_dir = _resolve_output_dir(
        ROOT, cfg.paths.preprocessed_dir, dataset_name, method
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = PREPROCESSOR_REGISTRY[method](cfg)
    preprocessor.run(output_dir)
    print(f"✓ Preprocessed {source}/{method} → {output_dir}")
