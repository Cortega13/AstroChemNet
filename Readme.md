### Note: This package is currently under development. Any and all suggestions for improvements are welcome.

## Surrogate Modeling for Astrochemical Networks.
This package contains training/testing procedures for training deep neural network surrogate models for astrochemical networks. We use the [UCLCHEM](https://github.com/uclchem/UCLCHEM) chemical network for our datasets. Datasets are available upon request. 

## Project Structure
```
AstroChemNet/
├── configs/                    # Training and evaluation configuration objects.
├── data/                       # Drop-in location for local .h5 datasets.
├── nn_architectures/           # PyTorch module definitions for the surrogate models.
├── plots/
│   ├── assets/                 # Static images used across documentation/notebooks.
│   └── trajectories/           # Generated trajectory visualisations.
├── research/                   # Exploratory notebooks testing alternative approaches.
├── scripts/
│   ├── analysis/               # Notebooks for post-training investigation.
│   ├── preprocessing/          # Data-prep notebooks and utilities.
│   └── train/                  # CLI entry points and helpers for training jobs.
├── src/
│   └── AstroChemNet/           # Installable package code.
│       ├── analysis.py         # Plotting and metrics helpers.
│       ├── data_loading.py     # Dataset loaders and batching helpers.
│       ├── data_processing.py  # Pre-/post-processing utilities.
│       ├── inference.py        # Inference pipelines.
│       ├── loss.py             # Loss definitions used during training.
│       ├── trainer.py          # High-level training loop.
│       └── utils.py            # Shared helper functions.
├── utils/                      # Cached numpy arrays and lookup tables.
├── vibecode/                   # ViBe baseline experiments and scripts.
│   └── archive/                # Historical utilities for ViBe experiments.
├── weights/
│   └── archived_original/      # Reference checkpoints and experiment artefacts.
├── .gitignore                  # Git ignore rules.
├── README.md                   # Project documentation (this file).
├── pyproject.toml              # Project configuration for packaging.
└── requirements.txt            # Python dependency lockfile for development.
```
## Setup Guide
Clone the repository.

```sh
git clone https://github.com/Cortega13/AstroChemNet.git
```

Enter the project directory.

```sh
cd AstroChemNet
```

Install the package using pip. Note: No necessary package dependencies are defined yet, so you must download them separately.

```sh
pip install -e .
```

## Gravitational Collapse Benchmark

## Turbulent Gas Benchmark
<!-- 
In order to validate the results of this model, you must download the dataset. The compressed version of this dataset is 9GB. You can download the dataset by downloading data.zip from the following Google Drive link.
[Click Here to Download Dataset](https://drive.google.com/file/d/1IuEMusFsvDlzqRHD4c5SMUoSv9CJ50gW/view?usp=drive_link)
You must then place data.zip inside the ChemSurrogate folder and extract it using your preferred method. This will create a data/ folder. Inside the data/ folder are two separate hdf5 files, one named uclchem_rawdata_training.h5 and the other uclchem_rawdata_validation.h5. The first holds 60k models and the second holds 20k models. Information on this dataset is discussed below.

If you are lucky and no bugs occured during the installation, then you may now refer to the examples/ folder for example scripts. Here lies the code which generated the plots in the plots/ folder.

## Training/Validation Dataset
The dataset is generated using UCLCHEM. Both datasets are sampled separately since Latin Hypercube Sampling is used. A dataset generation script is included in this repository to replicate results. The datasets use the following initial abundances and physical parameter ranges.

All abundances are normalized to H nuclei abundance.
![Initial Abundances](initial_abundances.jpeg)

These physical parameters are sampled in log space.
| Parameter  | Range (Min, Max) | Units | Notes |
|------------|----------------|--------|------------------------------------------------------------|
| **Density**  | (1e1, 1e6)    | H nuclei per cm³ | Limits arbitrarily chosen. |
| **Radfield** | (1e-3, 1e3)   | Habing field | Limits arbitrarily chosen. |
| **Av**       | (1e-2, 1e4)   | Magnitudes | Limits arbitrarily chosen. |
| **GasTemp**  | (10, 150)     | Kelvin | Grain reactions are too complex below 10K. Ice mostly sublimates at 150K, and UCLChem sets this as a strict constraint. |

![Physical Parameter Sampling](physical_parameter_sampling.png)

60k models were created for training and 20k for validation. The initial abundances are constant for all models, and only the physical parameters are varied. The physical parameters were sampled using latin hypercube sampling in log-space. The Cosmic Ray Ionization Rate is kept constant at 3.0261e-17 ionizations per second. 

With 100 timesteps per model, the total dataset size is 8,000,000 rows of data. Since the model is trained to predict 1-100 timesteps, we can train using all combinations of pairs of abundances. For example (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), ... etc

The validation loss for the results are defined as

loss = abs(actual - predicted) / actual

This loss is calculated for each species and then the mean is calculated. This is calculated for the entire validation set. The species abundances range from 1e-20 to 1. -->
