# Repetitive Tasks

## Training New Model Variant

1. Create YAML in `configs/models/` (e.g., `autoencoder_large.yaml`)
2. Train: `astrochemnet-train-autoencoder model=new_model_name`
3. Analyze in `scripts/models/Error Analysis Autoencoder.ipynb`

**Notes:** Ensure `input_dim` matches dataset (333 for abundances, 18 for emulator). Use `ModelsConfig` schema.

## Adding New Dataset

1. Preprocess: `astrochemnet-preprocess`
2. Create YAML in `configs/datasets/` with path, phys params, species file, clipping ranges
3. If species changed, regenerate stoichiometric matrix via `Processing.build_stoichiometric_matrix()`
4. Train: `astrochemnet-train-autoencoder dataset=new_dataset_name`

**Notes:** HDF5 structure = (Index, Model, Time, phys params, species). Update `input_dim` if species count changed.

## Hyperparameter Tuning

1. Preprocess once: `astrochemnet-train-autoencoder mode=preprocess`
2. Run experiments: `astrochemnet-train-autoencoder mode=train model.lr=1e-4 model.batch_size=32768`
3. Compare results in timestamped `outputs/` directories
4. Track best configs vs PCA+KNN baseline (0.057% MAPE)

## Error Analysis

1. Load model via `Inference` class
2. Predict on validation set
3. Compute errors: `calculate_relative_error()`
4. Check conservation violations, identify problematic species (sulfur chemistry)
5. Compare to KNN baseline

**Files:** `scripts/models/Error Analysis [Autoencoder|Emulator].ipynb`, `inference.py`, `analysis.py`

## Full Training Pipeline

```bash
astrochemnet-preprocess                  # Optional if data ready
astrochemnet-train-autoencoder           # → weights/autoencoder.pth, utils/latents_minmax.npy
astrochemnet-train-emulator              # → weights/mlp.pth (requires pretrained autoencoder)
```

Analyze in `scripts/models/Error Analysis Emulator.ipynb`. Pipeline takes hours to days.
