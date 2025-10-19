1. Goal
We are building a surrogate model for astrochemical simulations. The goal is to train machine learning models that can imitate the results of expensive chemistry simulations (done with UCLCHEM) so that we can predict chemical abundances much faster.

Data and Simulation Pipeline

2. Physical Simulation (Gravitational Collapse)
We start with a spherical gravitational collapse simulation that models how gas and dust evolve over time.
Tracers: about 9,900 Lagrangian tracers (points that follow the collapsing gas)
Timesteps: about 300 steps, spaced 92.9 years apart
Evolved parameters: density, visual extinction (Av), gas temperature, and radiation field
The resulting dataset shape is approximately (300 timesteps × 4 parameters × 9,900 tracers).

3. Chemistry Generation (UCLCHEM)
Next, we post-process each tracer using UCLCHEM, which simulates interstellar chemistry by solving large, stiff systems of ODEs.
This produces abundances for 333 chemical species at each timestep.
The results are stored in an HDF5 file that contains:
Metadata (data index, time, model number)
Physical parameters (density, temperature, etc.)
Chemical species abundances
This is our main dataset.

4. Data Preparation
Data preprocessing is done in scripts/preprocessing.
We split the dataset into training and validation sets, remove redundant entries, and clean up the data.
We also normalize the chemistry values:
1. Clip values to [1e-20, 1]
2. Apply log10 transform (range −20 to 0)
3. MinMax scale to [0, 1]
The cleaned and normalized datasets are saved as HDF5 files for reuse during training.

Modeling and Training

5. Autoencoder
We train an autoencoder in scripts/train to learn a compressed representation of the chemical abundances.
Input and output: 333 chemical species
Latent space: 14 components
Purpose: compress high-dimensional chemistry into a smaller latent vector
Note: the autoencoder does not use physical parameters, time, or tracer number; it only focuses on species patterns.
After training, we save both the normalization parameters and the encoded latent representations for each tracer.

6. Emulator
The emulator predicts how the latent chemical state evolves over time, given the physical conditions.
For each tracer, we create sliding windows of 240 timesteps from the total 300, producing about 60 samples per tracer.
The emulator takes the initial latent vector as input, predicts the next 240 latent steps autoregressively (feeding each output into the next step), and uses the full physical parameter sequence (density, temperature, etc.) as input to guide the evolution.
This allows us to reproduce the chemical evolution without running UCLCHEM directly.

Visualization and Analysis

7. Sequence Analysis (vibecode/)
In vibecode, we explore patterns and structure in the data.
We apply PCA (Principal Component Analysis) to reduce dimensionality and use t-SNE to visualize how different tracers or conditions group together.
This is done separately for the physical parameter sequences and the chemistry sequences.
