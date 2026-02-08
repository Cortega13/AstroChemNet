# Carbox gravitational collapse dataset (`carbox_grav`)

## What this is

This repo already has a UCLCHEM-style gravitational collapse dataset at [`data/gravitational_collapse.h5`](data/gravitational_collapse.h5:1) configured by [`configs/data/grav.yaml`](configs/data/grav.yaml:1). The Carbox dataset at [`data/carbox_gravitational_collapse.h5`](data/carbox_gravitational_collapse.h5:1) uses a different on-disk layout and column naming scheme, so it cannot be fed directly into the existing [`src.preprocessors.uclchem_grav.UclchemGravPreprocessor`](src/preprocessors/uclchem_grav.py:83).

To make it usable for training/validation, the workflow in [`scripts/carbox_dataset_analysis.py`](scripts/carbox_dataset_analysis.py:1) does three things:

- Reads and analyzes the Carbox HDF5 group layout.
- Normalizes column names into the repo’s canonical names.
- Writes train/val 3D tensors and the corresponding dataset config.

## HDF5 layout

`carbox_gravitational_collapse.h5` is not a pandas HDFStore like the grav dataset.

For the training split we use the `large` group:

- Group: `large`
- Datasets under `large/`:
  - `columns` (length = number of columns)
  - `data` (shape = `(n_rows, n_cols)`, dtype `float32`)
  - `tracer_id` (shape = `(n_rows,)`, dtype `int64`)
  - `tracer_ptr` (shape = `(n_tracers, 3)`, dtype `int64`)

`tracer_ptr[i] = [tracer_id, start_row, length]` describes the contiguous row block for a tracer.

## Observed dataset statistics (key: `large`)

These are produced by running [`scripts/carbox_dataset_analysis.py`](scripts/carbox_dataset_analysis.py:1) and are also written to [`outputs/analysis/carbox_grav/summary.json`](outputs/analysis/carbox_grav/summary.json:1).

- Rows (`n_rows`): `4,000,773`
- Columns (`n_cols`): `168`
- Tracers (`n_tracers`): `10,027`
- Tracer length (all tracers): `399` rows per tracer
- Time column range:
  - min: `100000.0`
  - max: `3403002.0`
- Physical parameter ranges (raw min/max):
  - `Density`: `[0.5715546011924744, 195699.671875]`
  - `Radfield`: `[3.8874722463333455e-07, 3.748945713043213]`
  - `Av`: `[0.0075920443050563335, 18.559925079345703]`
  - `gasTemp`: `[2.591064214706421, 7209.87451171875]`
- Species coverage:
  - master list size (repo): `333` from [`data/species.txt`](data/species.txt:1)
  - present in Carbox `large`: `163`
- Abundance extrema (raw):
  - min: `1.0000000195414814e-24`
  - max: `1.0`
  - count below clip lower bound (`1e-20`): `473,178,884`

Implication: clipping is mandatory for consistency with the rest of the pipeline (same rationale as in [`configs/data/grav.yaml`](configs/data/grav.yaml:28)).

## Column normalization and renaming

The Carbox column names are lowercase for physical parameters and include a few species names that do not match the repo’s canonical gas-phase naming.

The script normalizes columns by applying substring replacements defined in [`scripts/carbox_dataset_analysis._rename_mapping()`](scripts/carbox_dataset_analysis.py:165).

Important mappings:

- Physical parameters:
  - `time` → `Time`
  - `density` → `Density`
  - `temperature` → `gasTemp`
  - `av` → `Av`
  - `rad_field` → `Radfield`

- Species alias fixes (examples):
  - `CH2CO` → `C2H2O`
  - `CH2OH` → `CH3O`
  - `CH3CCH` → `C3H4`
  - `CH3CHO` → `C2H4O`
  - `CH3CN` → `C2H3N`
  - `CH3OH` → `CH4O`
  - `NH2CHO` → `CH3NO`
  - `HCOOH` → `H2CO2`
  - `H2COH+` → `H3CO+`
  - `H2CSH+` → `H3CS+`
  - `HOSO+` → `HSO2+`
  - `OCSH+` → `HOCS+`
  - `SISH+` → `HSIS+`
  - `NCCN` → `N2C2`

The dataset contains no duplicate columns after renaming for `large`.

## Why only 163 species

[`data/species.txt`](data/species.txt:1) includes gas-phase species plus surface (`#...`) and bulk (`@...`) species. The Carbox `large` group contains only a subset of the gas-phase species plus the physical columns.

The script writes a dataset-specific species file containing only the species actually present:

- [`data/carbox_grav_species.txt`](data/carbox_grav_species.txt:1)

This is what the dataset config uses.

## Train/validation preparation

The output format matches what the rest of the repo expects after “uclchem_grav” preprocessing: a 3D float32 tensor shaped like:

- `(n_tracers, n_timesteps + 1, n_features)`

For this dataset:

- `n_tracers = 10027`
- `n_timesteps = 399` (from `tracer_ptr[:, 2]`)
- an extra initial row is prepended, so tensor time dimension is `400`
- `n_features = 4 + 163 = 167`

The initial row is constructed as:

- physical params set to `0`
- abundances set from [`data/initial_abundances.npy`](data/initial_abundances.npy:1), aligned by species name

Then the same “physical shift” logic used by [`src.preprocessors.uclchem_grav.UclchemGravPreprocessor.run()`](src/preprocessors/uclchem_grav.py:225) is applied: physical parameters are shifted by -1 timestep so that each state has the physical conditions of the next time step.

### Split strategy

The split is done by tracer (not by row):

- `train_split = 0.75`
- `seed = 42`

Resulting counts (from [`outputs/analysis/carbox_grav/prepare.json`](outputs/analysis/carbox_grav/prepare.json:1)):

- train tracers: `7520`
- val tracers: `2507`

## Generated files

Running the script creates:

- Analysis outputs:
  - [`outputs/analysis/carbox_grav/summary.json`](outputs/analysis/carbox_grav/summary.json:1)
  - [`outputs/analysis/carbox_grav/summary.txt`](outputs/analysis/carbox_grav/summary.txt:1)
  - [`outputs/analysis/carbox_grav/summary.md`](outputs/analysis/carbox_grav/summary.md:1)
  - [`outputs/analysis/carbox_grav/prepare.json`](outputs/analysis/carbox_grav/prepare.json:1)

- Dataset config:
  - [`configs/data/carbox_grav.yaml`](configs/data/carbox_grav.yaml:1)

- Dataset-specific species list:
  - [`data/carbox_grav_species.txt`](data/carbox_grav_species.txt:1)

- Preprocessed tensors and stoichiometry:
  - [`outputs/preprocessed/carbox_grav/uclchem_grav/uclchem_grav_train_preprocessed.pt`](outputs/preprocessed/carbox_grav/uclchem_grav/uclchem_grav_train_preprocessed.pt:1)
  - [`outputs/preprocessed/carbox_grav/uclchem_grav/uclchem_grav_val_preprocessed.pt`](outputs/preprocessed/carbox_grav/uclchem_grav/uclchem_grav_val_preprocessed.pt:1)
  - [`outputs/preprocessed/carbox_grav/uclchem_grav/stoichiometric_matrix.pt`](outputs/preprocessed/carbox_grav/uclchem_grav/stoichiometric_matrix.pt:1)

## Migration note

- Existing artifacts under `outputs/preprocessed/*/initial/` are obsolete after the stage rename.
- Regenerate preprocessing outputs to create `outputs/preprocessed/*/uclchem_grav/` artifacts.

## How to run

To reproduce everything:

- `python scripts/carbox_dataset_analysis.py`

Common switches (all defined in [`scripts/carbox_dataset_analysis.py`](scripts/carbox_dataset_analysis.py:72)):

- `--skip-prepare` to only compute stats and write configs/species lists.
- `--chunksize <N>` to trade memory vs speed.

## Training notes

The existing component configs (for example [`configs/components/autoencoder_grav.yaml`](configs/components/autoencoder_grav.yaml:1)) are set up for `dataset: grav` and expect `333` abundance features.

For Carbox you will need a new component config with:

- `dataset: carbox_grav`
- model input sizes consistent with `n_species = 163`

The training entrypoint is [`train.py`](train.py:1), which loads a component config by name and then loads the dataset config indicated by `component.dataset`.

## Gotchas / caveats

- The Carbox dataset does not include surface/bulk species (`#...` / `@...`) from the repo master species list, so any model expecting those channels will need to be resized.
- The raw dataset contains many abundances below `1e-20`; these are clipped to `1e-20` before writing tensors (consistent with the clipping configuration used elsewhere).
- The Carbox `large` format does not contain an explicit `Model` column; tracers are defined by `tracer_ptr` blocks and the split is performed at the tracer level.
