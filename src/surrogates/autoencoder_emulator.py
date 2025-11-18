from .base_surrogate import BaseSurrogate


class AutoencoderEmulatorSurrogate(BaseSurrogate):
    def benchmark(self):
        # TODO: Implement autoencoder + emulator benchmarking
        return {"type": "ae_emulator", "results": "placeholder"}
