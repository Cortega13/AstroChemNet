# Repetitive Tasks

This file documents common repetitive workflows for future reference.

---

## Training New Model Variant

**When to use:** Creating a new model architecture or hyperparameter configuration

**Files to modify:**
- Create new config in `configs/models/` (e.g., `autoencoder_large.yaml` or `emulator_deep.yaml`)
- Optionally modify model class in `src/AstroChemNet/models/` if architecture changes

**Steps:**
1. Create new YAML config in `configs/models/`
2. Define architecture parameters (layers, dimensions, dropout, etc.)
3. Train using CLI with model override: `astrochemnet-train-autoencoder model=new_model_name`
4. Monitor training metrics in `outputs/` directory (Hydra creates timestamped subdirs)
5. Analyze results in Jupyter notebooks (`scripts/models/Error Analysis Autoencoder.ipynb`)

**Important notes:**
- Model config must use `ModelsConfig` schema fields
- Ensure `input_dim` matches dataset (333 for abundances, 18 for emulator=4+14)
- Test with small batch first to verify config correctness
- Check `outputs/` for training logs and metrics JSON files

---

## Adding New Dataset

**When to use:** Training on different physical simulations (e.g., turbulent flows, different environments)

**Files to modify:**
- Create new config in `configs/datasets/` (e.g., `turbulent.yaml`)
- Add HDF5 file to `data/` directory
- Update `utils/` if species list changes

**Steps:**
1. Preprocess raw UCLCHEM data: `astrochemnet-preprocess`
2. Create dataset YAML config specifying:
   - Dataset path (preprocessed HDF5)
   - Physical parameter ranges (Density, Radfield, Av, gasTemp)
   - Species file path (if different from default)
   - Abundances clipping ranges
3. If species changed, regenerate stoichiometric matrix using `Processing.build_stoichiometric_matrix()`
4. Train with dataset override: `astrochemnet-train-autoencoder dataset=new_dataset_name`
5. Update model `input_dim` if number of species changed

**Important notes:**
- HDF5 must have expected structure (Index, Model, Time, phys params, species)
- Physical parameters should be in log space
- Species list in `utils/species.txt` must match dataset
- Use `metadata: [Index, Model, Time]` and `phys: [Density, Radfield, Av, gasTemp]` fields

---

## Hyperparameter Tuning Experiment

**When to use:** Optimizing model performance through systematic parameter search

**Files to modify:**
- None (use CLI overrides)

**Steps:**
1. Preprocess dataset once: `astrochemnet-train-autoencoder mode=preprocess`
2. Run training experiments with different hyperparameters:
   ```bash
   astrochemnet-train-autoencoder mode=train model.lr=1e-4 model.batch_size=32768
   astrochemnet-train-autoencoder mode=train model.lr=5e-4 model.batch_size=16384
   astrochemnet-train-autoencoder mode=train model.dropout=0.3 model.latent_dim=12
   ```
3. Compare results in `outputs/` directories (timestamped)
4. Analyze with notebooks in `scripts/models/`
5. Check JSON files for per-epoch metrics

**Important notes:**
- Use `mode=preprocess` once, then `mode=train` for all experiments
- Each run creates timestamped output directory under `outputs/`
- Track best performing configs manually or in separate log
- Compare against PCA+KNN baseline (0.057% MAPE)

---

## Error Analysis Workflow

**When to use:** Diagnosing model performance issues, identifying problematic species

**Files to use:**
- `scripts/models/Error Analysis Autoencoder.ipynb`
- `scripts/models/Error Analysis Emulator.ipynb`
- `src/AstroChemNet/inference.py`
- `src/AstroChemNet/analysis.py`

**Steps:**
1. Load trained model using `Inference` class
2. Run predictions on validation set
3. Compute per-species errors using `calculate_relative_error()`
4. Visualize error distributions (histograms, scatter plots)
5. Identify species with high errors (sulfur chemistry typically challenging)
6. Check conservation constraint violations
7. Examine physical parameter correlations
8. Compare against KNN baseline performance

**Important notes:**
- Use validation set, not training set
- Check both reconstruction error (autoencoder) and prediction error (emulator)
- Elemental conservation is critical physical constraint
- KNN baseline achieves 0.057% mean error - neural models should match or exceed
- Sulfur-bearing species (S, HS2, S+, OCS) often have higher errors

---

## Full Training Pipeline (Autoencoder → Emulator)

**When to use:** Training complete surrogate model from scratch

**Steps:**
1. Preprocess raw UCLCHEM data (if needed):
   ```bash
   astrochemnet-preprocess
   ```

2. Train autoencoder:
   ```bash
   astrochemnet-train-autoencoder
   ```
   - Saves weights to `weights/autoencoder.pth`
   - Saves latent min/max to `utils/latents_minmax.npy`

3. Train emulator:
   ```bash
   astrochemnet-train-emulator
   ```
   - Requires pretrained autoencoder
   - Saves weights to `weights/mlp.pth`

4. Analyze results:
   - Open `scripts/models/Error Analysis Emulator.ipynb`
   - Compare predictions vs ground truth
   - Check conservation errors
   - Compare to KNN baseline

**Important notes:**
- Autoencoder must be trained before emulator
- Emulator automatically loads pretrained autoencoder for inference
- Each stage creates timestamped output directory
- Full pipeline can take hours to days depending on hardware

---

*Add new tasks here as they are discovered and completed*
