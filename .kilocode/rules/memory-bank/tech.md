# Technology Stack

## Core Framework

**PyTorch** - Deep learning framework
- CUDA 12.6 support for GPU acceleration
- Custom dataset classes with `__getitems__` batching
- Automatic device detection (CUDA/CPU)
- JIT compilation for performance-critical functions

## Configuration Management

**Hydra** - Structured configuration system
- YAML-based configs with composition
- CLI overrides: `model.lr=1e-4 dataset=grav`
- Structured config validation via dataclasses
- Automatic working directory management

## Data Handling

**HDF5** (via h5py, tables)
- Time-series storage: 9989 tracers × 298 timesteps
- Datasets: physical parameters (4-D) and abundances (333-D)
- Efficient chunked reading for large datasets

**NumPy**
- Array operations and transformations
- Stoichiometric matrix operations
- Physical parameter scaling

**Pandas**
- Metrics logging to CSV/JSON
- Data analysis in notebooks
- HDF5 loading via `pd.read_hdf()`

## Scientific Computing

**SciPy**
- Statistical operations
- Optimization utilities

**Numba**
- JIT compilation for performance-critical code
- Used in `calculate_emulator_indices()` function

## Visualization

**Matplotlib** - Primary plotting library
- Loss curves, error distributions
- Abundance comparisons

**Seaborn** - Statistical visualizations
- Enhanced aesthetics for analysis plots

## Development Tools

**Ruff** - Fast Python linter and formatter
- Replaces flake8, black, isort
- Commands:
  - `ruff check --fix` - lint and auto-fix
  - `ruff format` - format code
- Configured in `pyproject.toml`

**Jupyter** - Interactive development
- Notebooks in `scripts/` and `research/`
- Error analysis and visualization

## Package Management

**pip** - Package installation
- `pip install -e .` - editable install
- `requirements.txt` for dependencies (may have encoding issues)
- `pyproject.toml` for package metadata

**setuptools** - Package building
- Entry points for CLI commands in `pyproject.toml`
- Automatic CLI registration: `astrochemnet-train-autoencoder`, `astrochemnet-train-emulator`, `astrochemnet-preprocess`

## Development Setup

### Installation

```bash
# Clone and navigate to project
cd AstroChemNet

# Install in editable mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

```

### Environment

- **Python 3.7+** (requires-python >= 3.7 in pyproject.toml)
- **CUDA 12.6** for GPU training
- **Windows 11** development environment (based on paths)

## Key Dependencies

```
torch>=2.0.0          # Deep learning
hydra-core>=1.3.0     # Configuration
h5py                  # HDF5 file I/O
tables                # Alternative HDF5 interface
numpy                 # Array operations
pandas                # Data analysis
scipy                 # Scientific computing
matplotlib            # Plotting
seaborn               # Statistical viz
tqdm                  # Progress bars
numba                 # JIT compilation
```

## Technical Constraints

### Memory Management
- Large datasets (9989 × 298 samples) require chunked processing
- `ChunkedShuffleSampler` for efficient data shuffling
- Preprocessed data cached to disk to avoid recomputation
- `pin_memory=True` for faster GPU transfer

### Numerical Stability
- Abundances clipped to [1e-20, 1.0] to prevent log(0)
- Batch normalization for stable training
- Gradient clipping in trainer (2.0 for autoencoder, 1.0 for emulator)

### Performance Optimizations
- `__getitems__` batch loading (10³× speedup over individual loads)
- Tied weights reduce autoencoder parameters by ~50%
- `num_workers=10` for DataLoader parallelism
- `in_order=False` for DataLoader flexibility
- `cudnn.benchmark=True` for performance
- TF32 enabled for matmul operations

## Configuration Patterns

### Hydra Decorators

```python
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Access nested configs
    device = cfg.dataset.device
    lr = cfg.model.lr
```

### CLI Overrides

```bash
# Override single parameter
astrochemnet-train-autoencoder model.lr=5e-4

# Override multiple parameters
astrochemnet-train-emulator model.batch_size=16384 model.dropout=0.3

# Change dataset/model configs
astrochemnet-train-autoencoder dataset=turbulent model=autoencoder_large

# Control preprocessing
astrochemnet-train-autoencoder mode=preprocess  # Only preprocess
astrochemnet-train-autoencoder mode=train       # Only train (load cached)
astrochemnet-train-autoencoder mode=both        # Default: both
```

## File Paths

### Data Locations
- Raw datasets: `data/gravitational_collapse.h5`
- Preprocessed datasets: `outputs/temp_data/grav_collapse_clean.h5`
- Preprocessed autoencoder: `data/autoencoder_train_preprocessed.pt`, `data/autoencoder_val_preprocessed.pt`
- Preprocessed emulator: `data/training_seq.h5`, `data/validation_seq.h5`
- Static arrays: `utils/` (species.txt, stoichiometric_matrix.npy, initial_abundances.npy, latents_minmax.npy)

### Model Weights
- Autoencoder: `weights/autoencoder.pth`
- Emulator: `weights/mlp.pth`
- Archived reference: `weights/archived_original/`

### Outputs
- Training logs: `outputs/` (Hydra creates timestamped subdirs)
- Metrics: JSON files in output directories

## Tool Usage Patterns

### Code Quality

```bash
# Auto-fix linting issues
ruff check --fix

# Format all Python files
ruff format

# Format specific notebook
ruff format scripts/models/Error\ Analysis.ipynb
```

### Training

```bash
# Standard training
astrochemnet-train-autoencoder
astrochemnet-train-emulator

# Hyperparameter tuning
astrochemnet-train-autoencoder model.lr=1e-4 model.batch_size=32768

# Different configurations
astrochemnet-train-emulator dataset=grav model=emulator
```

### Development Workflow

1. Edit code in `src/AstroChemNet/`
2. Reinstall if CLI modified: `pip install -e . --force-reinstall --no-deps`
3. Run training with overrides
4. Analyze in Jupyter notebooks (`scripts/models/`)
5. Format and lint before commit


## Important Notes

- Hydra automatically creates timestamped output directories under `outputs/`
- Config uses `${working_path}` variable resolved at runtime
- Dataset config uses `phys` field (not `parameters` as shown in some files) - there's an inconsistency to note
