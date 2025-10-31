# AstroChemNet - Quick Reference

## What It Is

Neural network surrogate models for astrochemical simulations. Replaces slow ODE integration (thousands of coupled equations for 333 species) with fast ML inference.

## Architecture

Two-stage pipeline:
1. **Autoencoder** - Compress 333-D abundances → 12-14D latent space
2. **Emulator** - Predict latent evolution given physical conditions (density, temperature, radiation, extinction)

## Essential Commands

```bash
# Setup
pip install -e .

# Training
astrochemnet-train-autoencoder              # Train compression model
astrochemnet-train-emulator                 # Train dynamics predictor
astrochemnet-train-autoencoder model.lr=1e-4  # Override config

# Code quality
ruff check --fix && ruff format
```

## Key Metrics

# We primarily care about the chemical abundances predictions metrics. These metrics muse use MAPE on unscaled abundances. Not log10.

- **Dataset**: 9,989 tracers × 298 timesteps (~3M samples)
- **Species range**: 10⁻²⁰ to 1.0 (20 orders of magnitude)
- **Challenging species**: S, HS₂, S⁺, OCS (sulfur chemistry)

## Project Status

Mature research codebase with working models. Active focus: error analysis, performance optimization, baseline comparison. Use CLI commands (not deprecated `scripts/train/`).

## Code Style

Functions ≤25 lines, type hints, one-line docstrings, minimalistic.
