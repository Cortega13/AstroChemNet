### Note: This package is currently under development. Any and all suggestions for improvements are welcome.

## Surrogate Modeling for Astrochemical Networks
This package contains training/testing procedures for deep neural surrogate models for astrochemical networks using [UCLCHEM](https://github.com/uclchem/UCLCHEM).

## Grav: super-concise train + benchmark

```sh
# 1) Preprocess grav data
python preprocess.py grav initial
python preprocess.py initial autoencoder
python preprocess.py initial autoregressive

# 2) Train models
python train.py component=autoencoder_grav
python train.py component=emulator_grav

# 3) Benchmark combined AE + autoregressive surrogate
python benchmark.py ae_emulator_grav
```

Weights are saved under `outputs/weights/` for each component model.
