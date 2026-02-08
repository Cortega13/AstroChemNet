### Note: This package is currently under development. Any and all suggestions for improvements are welcome.

## Surrogate Modeling for Astrochemical Networks
This package contains training/testing procedures for deep neural surrogate models for astrochemical networks using [UCLCHEM](https://github.com/uclchem/UCLCHEM).

## Grav: super-concise train + benchmark

```sh
# 1) Preprocess grav data
python run.py preprocess uclchem_grav initial
python run.py preprocess initial autoencoder
python run.py preprocess initial autoregressive

# 2) Train models
python run.py train autoencoder_grav
python run.py train emulator_grav

# 3) Benchmark combined AE + autoregressive surrogate
python run.py benchmark ae_emulator_grav
```

Weights are saved under `outputs/weights/` for each component model.
