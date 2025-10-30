# Product Description

## Purpose

AstroChemNet accelerates astrochemical network simulations by replacing computationally expensive ODE integrations with deep learning surrogate models. Enables large-scale 3D hydrodynamic simulations with full chemistry by providing orders-of-magnitude speedup while maintaining physical accuracy.

## Problem Being Solved

Astrochemical networks model 333 chemical species with abundances spanning 20 orders of magnitude (10⁻²⁰ to 1.0 relative to H nuclei). Direct integration of thousands of coupled stiff ODEs describing gas-phase reactions, grain surface chemistry, freeze-out, and desorption is prohibitively slow for production simulations with millions of cells.

Traditional UCLCHEM integration cannot scale to multi-dimensional hydrodynamic simulations.

## Solution Approach

## Training Data Workflow

1. Run hydrodynamic simulation (e.g., gravitational collapse)
2. Place ~10,000 Lagrangian tracer particles in simulation
3. Extract physical trajectories (density, temperature, radiation field, visual extinction)
4. Post-process chemistry: run UCLCHEM for each tracer's trajectory
5. Store time-series data in HDF5 format

## Deployment Workflow

1. Train autoencoder on chemical abundances to learn compressed representation
2. Encode training data to latent space
3. Train emulator to predict latent space evolution given physical parameters
4. Deploy: encode → emulator step → decode for fast chemistry prediction
