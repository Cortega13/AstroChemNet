### Note: This package is currently under development.

## Surrogate Modeling for Astrochemical Networks
This package contains training, preprocessing, and benchmarking workflows for deep neural surrogate models for astrochemical networks using [UCLCHEM](https://github.com/uclchem/UCLCHEM).

## Configuration-driven workflow
Runtime behavior is configured from in-code dataclass registries in `src/configs/loader.py`.

### Available configuration names
- **Datasets**
  - `uclchem_grav`
  - `carbox_grav`
- **Preprocessing methods**
  - `uclchem_grav`
  - `autoencoder`
  - `autoregressive`
- **Training components**
  - `autoencoder_uclchem_grav`
  - `emulator_uclchem_grav`
- **Benchmark surrogates**
  - `autoencoder_emulator_uclchem_grav`

## Quick start (grav pipeline)

```sh
# 1) Preprocess data (shorthand targets)
python run.py preprocess uclchem_grav
python run.py preprocess autoencoder_uclchem_grav
python run.py preprocess emulator_uclchem_grav

# 2) Train component models
python run.py train autoencoder_uclchem_grav
python run.py train emulator_uclchem_grav

# 3) Benchmark the configured surrogate bundle
python run.py benchmark autoencoder_emulator_uclchem_grav
```

## CLI summary

```sh
python run.py preprocess <dataset_or_component>
python run.py train <component>
python run.py benchmark <surrogate>
```

Preprocess dependency rules:
- `python run.py preprocess <dataset>` runs `uclchem_grav` preprocessing for that dataset.
- `python run.py preprocess <autoencoder_component>` runs `autoencoder` preprocessing and errors if `uclchem_grav` preprocessing is missing.
- `python run.py preprocess <emulator_component>` runs `autoregressive` preprocessing and errors if `uclchem_grav` preprocessing or pretrained autoencoder weights are missing.

## Migration note
- Old preprocessing artifacts under `outputs/preprocessed/*/initial/` are obsolete.
- Old tensor files named `initial_train_preprocessed.pt` and `initial_val_preprocessed.pt` are obsolete.
- Regenerate preprocessing outputs so artifacts are recreated under the `uclchem_grav` stage naming.

## Outputs
- Preprocessed tensors: `outputs/preprocessed/<dataset>/<method>/`
- Model artifacts: `outputs/weights/<component>/`
- Trainer exports include `weights.pth`, `config.json`, and `metrics.json`

Using explicit config names in CLI calls is recommended so runs stay aligned with the registry definitions.
