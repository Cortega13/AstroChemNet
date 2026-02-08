"""Defines surrogate names and class resolution helpers."""

from enum import StrEnum

from src.surrogates.autoencoder_emulator import AutoencoderEmulatorSurrogate


class SurrogateName(StrEnum):
    """Enumerates supported surrogate names."""

    AE_EMULATOR_GRAV = "ae_emulator_grav"

    @classmethod
    def from_name(cls, name: str) -> "SurrogateName":
        """Converts a surrogate name string into the enum value."""
        return cls(name)

    def to_class(self) -> type[AutoencoderEmulatorSurrogate]:
        """Converts the enum value into the surrogate class."""
        if self is SurrogateName.AE_EMULATOR_GRAV:
            return AutoencoderEmulatorSurrogate
        raise ValueError(f"Unsupported surrogate name: {self}")
