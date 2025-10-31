# KNN Analysis Report: `plot-knn-analysis.py`

## Overview
`scripts/datasets/knn/plot-knn-analysis.py` is a comprehensive analysis script that evaluates KNN-based trajectory predictions for astrochemical simulations. It uses PCA dimensionality reduction and KNN regression to predict chemical abundance trajectories from physical conditions.

## Core Functionality

### Data Processing Pipeline
1. **Data Loading**: Loads HDF5 trajectory data with 333 chemical species and 4 physical parameters
2. **Trajectory Extraction**: Groups data by model ID, converts abundances to log₁₀ space
3. **PCA Compression**: Reduces high-dimensional trajectory data (333×298 features) to 5-component representation
4. **KNN Prediction**: Distance-weighted averaging in PCA space for trajectory prediction

### Error Analysis
- **MAPE Calculation**: Mean Absolute Percentage Error for species abundances
- **Per-Species Errors**: Individual error statistics for each chemical species
- **Trajectory-Level Errors**: Overall prediction quality per validation trajectory
- **Temporal Analysis**: Error evolution across simulation timesteps

### Visualization Suite
- **t-SNE Manifolds**: 2D embeddings showing trajectory distribution and error patterns
- **Trajectory Examples**: Side-by-side comparison of predicted vs. actual chemical evolution
- **Error Distributions**: Statistical analysis of prediction errors
- **Parameter Evolution**: Physical conditions in log₁₀ scale (density, radiation, extinction, temperature)

## Key Technical Details

### Data Structure
- **Input**: 9,989 trajectories × 298 timesteps × (333 species + 4 parameters)
- **PCA**: 98.6% variance explained with 5 components
- **KNN**: 10-neighbor distance-weighted averaging

### Error Metrics
- **MAPE Calculation**: Computed on unscaled (linear) abundances, not log-transformed values
- Mean MAPE: ~1.46% across all trajectories
- Best trajectory: <0.5% MAPE
- Worst trajectory: ~58% MAPE
- Top problematic species: @HE (6.39%), SIH₃ (5.24%), SIH₄ (5.14%)

### Output Files
- `tsne_manifold_errors.png`: Error visualization in trajectory manifold
- `trajectory_examples/`: Individual trajectory comparisons (best/median/worst)
- `mape_vs_timestep.png`: Temporal error evolution
- `species_error_statistics.json`: Detailed per-species error analysis

## Recent Fixes
- Corrected physical parameter scaling (removed erroneous `10**` transformation)
- Added log₁₀ display for physical parameters
- Removed KNN predictions for physical conditions (inputs, not predictions)

## Performance Characteristics
- Processes ~3M data points efficiently
- PCA reconstruction artifacts excluded from timestep 0 analysis
- Handles 20-order-of-magnitude abundance ranges
- Robust to negative values from PCA reconstruction

## Usage
```bash
cd scripts/datasets/knn
python plot-knn-analysis.py
```

Outputs saved to `outputs/plots/knn_analysis/` with comprehensive error analysis and visualizations.
