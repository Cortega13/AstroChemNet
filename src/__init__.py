"""Top-level package API for CLI orchestration."""

from pathlib import Path

from src.datasets import (
    DatasetName,
    build_dataset_config,
    get_preprocessor,
    list_available_datasets,
)


def ensure_dataset_preprocessed(
    dataset_name: str | DatasetName,
    force: bool = False,
    max_rows: int | None = None,
) -> None:
    """Build dataset preprocessing artifacts when missing or forced."""
    dataset_name = DatasetName(dataset_name)
    species_path = Path(__file__).resolve().parents[1] / "outputs" / dataset_name / "dataset" / "species.json"
    if not force and species_path.exists():
        return
    preprocessor = get_preprocessor(dataset_name)
    if preprocessor is None:
        raise KeyError(f"Unknown dataset: {dataset_name}")
    print(f"Preprocessing dataset: {dataset_name}")
    preprocessor(dataset_name=dataset_name, force=force, max_rows=max_rows)
    print(f"Preprocessing complete for: {dataset_name}")


def handle_train(
    model: str,
    dataset_name: str | DatasetName = DatasetName.UCLCHEM_GRAV,
    force_preprocess: bool = False,
) -> None:
    """Handle train command for a model and dataset."""
    from src.models import MODEL_REGISTRY, ModelName

    model = ModelName(model)
    ensure_dataset_preprocessed(dataset_name, force=force_preprocess)
    dataset_config = build_dataset_config(dataset_name)
    print(f"Training {model} on {dataset_config.device}")
    MODEL_REGISTRY[model].train(
        dataset_config,
        force_preprocess=force_preprocess,
    )


def handle_preprocess(
    dataset: str | DatasetName,
    force: bool = False,
    max_rows: int | None = None,
) -> None:
    """Handle preprocess command for dataset preparation."""
    dataset = DatasetName(dataset)
    preprocessor = get_preprocessor(dataset)
    if preprocessor:
        print(f"Preprocessing dataset: {dataset}")
        preprocessor(dataset_name=dataset, force=force, max_rows=max_rows)
        print(f"Preprocessing complete for: {dataset}")
    else:
        print(f"Unknown dataset: {dataset}")
        print(f"Available datasets: {', '.join(list_available_datasets())}")


def handle_benchmark(
    model: str,
    dataset_name: str | DatasetName = DatasetName.UCLCHEM_GRAV,
) -> None:
    """Handle benchmark command for model evaluation."""
    from src.models import MODEL_REGISTRY, ModelName

    model = ModelName(model)
    ensure_dataset_preprocessed(dataset_name)
    dataset_config = build_dataset_config(dataset_name)
    print(f"Benchmarking {model}...")
    results = MODEL_REGISTRY[model].benchmark(dataset_config)
    print(f"{model} Results: {results}")


def __getattr__(name: str):
    """Lazily expose model registry objects from src.models."""
    if name == "MODEL_REGISTRY":
        from src.models import MODEL_REGISTRY

        return MODEL_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "ensure_dataset_preprocessed",
    "handle_benchmark",
    "handle_preprocess",
    "handle_train",
]
