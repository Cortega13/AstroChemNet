"""Preprocessor registry."""

from .abundances_only import AbundancesOnlyPreprocessor
from .initial import InitialPreprocessor
from .markovianautoregressive import MarkovianautoregressivePreprocessor

PREPROCESSOR_REGISTRY = {
    "initial": InitialPreprocessor,
    "abundances_only": AbundancesOnlyPreprocessor,
    "markovianautoregressive": MarkovianautoregressivePreprocessor,
}
