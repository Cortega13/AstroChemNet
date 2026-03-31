"""Model registry and config builders."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.models.autoencoder.benchmark import benchmark as benchmark_autoencoder
from src.models.autoencoder.config import AEConfig
from src.models.autoencoder.config import build_config as build_autoencoder_config
from src.models.autoencoder.train import train as train_autoencoder
from src.models.autoregressive.benchmark import benchmark as benchmark_autoregressive
from src.models.autoregressive.config import build_config as build_autoregressive_config
from src.models.autoregressive.train import train as train_autoregressive
from src.models.latent_autoregressive.benchmark import (
    benchmark as benchmark_latent_autoregressive,
)
from src.models.latent_autoregressive.benchmark import (
    benchmark_combined_pipeline,
)
from src.models.latent_autoregressive.config import (
    build_config as build_latent_autoregressive_config,
)
from src.models.latent_autoregressive.train import train as train_latent_autoregressive
from src.models.latent_neural_operator.benchmark import (
    benchmark as benchmark_latent_neural_operator,
)
from src.models.latent_neural_operator.config import (
    build_config as build_latent_neural_operator_config,
)
from src.models.latent_neural_operator.train import (
    train as train_latent_neural_operator,
)
from src.models.latent_ode.benchmark import benchmark as benchmark_latent_ode
from src.models.latent_ode.config import build_config as build_latent_ode_config
from src.models.latent_ode.train import train as train_latent_ode
from src.models.latent_rnn.benchmark import benchmark as benchmark_latent_rnn
from src.models.latent_rnn.config import build_config as build_latent_rnn_config
from src.models.latent_rnn.train import train as train_latent_rnn


@dataclass(frozen=True)
class ModelEntry:
    """Model registry entry for CLI dispatch."""

    build_config: Callable[..., Any]
    train: Callable[..., None]
    benchmark: Callable[..., dict]


class ModelName(StrEnum):
    """Model names for registry lookup."""

    AUTOENCODER = "autoencoder"
    AUTOREGRESSIVE = "autoregressive"
    LATENT_AUTOREGRESSIVE = "latent_autoregressive"
    LATENT_RNN = "latent_rnn"
    LATENT_NEURAL_OPERATOR = "latent_neural_operator"
    LATENT_ODE = "latent_ode"
    COMBINED = "combined"


MODEL_REGISTRY = {
    ModelName.AUTOENCODER: ModelEntry(
        build_config=build_autoencoder_config,
        train=train_autoencoder,
        benchmark=benchmark_autoencoder,
    ),
    ModelName.AUTOREGRESSIVE: ModelEntry(
        build_config=build_autoregressive_config,
        train=train_autoregressive,
        benchmark=benchmark_autoregressive,
    ),
    ModelName.LATENT_AUTOREGRESSIVE: ModelEntry(
        build_config=build_latent_autoregressive_config,
        train=train_latent_autoregressive,
        benchmark=benchmark_latent_autoregressive,
    ),
    ModelName.LATENT_RNN: ModelEntry(
        build_config=build_latent_rnn_config,
        train=train_latent_rnn,
        benchmark=benchmark_latent_rnn,
    ),
    ModelName.LATENT_NEURAL_OPERATOR: ModelEntry(
        build_config=build_latent_neural_operator_config,
        train=train_latent_neural_operator,
        benchmark=benchmark_latent_neural_operator,
    ),
    ModelName.LATENT_ODE: ModelEntry(
        build_config=build_latent_ode_config,
        train=train_latent_ode,
        benchmark=benchmark_latent_ode,
    ),
    ModelName.COMBINED: ModelEntry(
        build_config=build_latent_autoregressive_config,
        train=train_latent_autoregressive,
        benchmark=benchmark_combined_pipeline,
    ),
}

AVAILABLE_MODELS: list[ModelName] = [
    ModelName.AUTOENCODER,
    ModelName.AUTOREGRESSIVE,
    ModelName.LATENT_AUTOREGRESSIVE,
    ModelName.LATENT_RNN,
    ModelName.LATENT_NEURAL_OPERATOR,
    ModelName.LATENT_ODE,
]
AVAILABLE_BENCHMARK_MODELS: list[ModelName] = [*AVAILABLE_MODELS, ModelName.COMBINED]


def build_model_config(
    model_name: str | ModelName,
    dataset_config,
    ae_config: AEConfig | None = None,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Build a model config for a dataset."""
    model_name = ModelName(model_name)
    entry = MODEL_REGISTRY[model_name]
    kwargs = {} if overrides is None else dict(overrides)
    if ae_config is None:
        return entry.build_config(dataset_config=dataset_config, **kwargs)
    return entry.build_config(dataset_config=dataset_config, ae_config=ae_config, **kwargs)


__all__ = [
    "AVAILABLE_BENCHMARK_MODELS",
    "AVAILABLE_MODELS",
    "MODEL_REGISTRY",
    "ModelEntry",
    "ModelName",
    "build_model_config",
]
