# Research Log

## KNN-Based Trajectory Prediction in PCA Space (2025-10-28)

### Motivation

Explored using K-Nearest Neighbors (KNN) in PCA-reduced space as a baseline for predicting chemical abundance trajectories. This approach tests whether the trajectory manifold has sufficient structure that simple interpolation between similar trajectories can provide reasonable predictions.

### Methodology

#### Data Structure
- Each trajectory represents one astrophysical model's chemical evolution
- **Trajectory shape**: 297 timesteps � 337 features
  - 333 chemical species abundances (log�� transformed)
  - 4 physical parameters: log��(density), log��(radfield), log��(Av), log��(gasTemp)
- **Flattening**: 297 � 337 = **100,089 features per trajectory**
- **Dataset split**:
  - Training: ~8,000 trajectories
  - Validation: 2,495 trajectories

#### Algorithm Pipeline

1. **Extract trajectories**: Combine log abundances + physical parameters for each model
2. **Flatten**: Convert (297 � 337) matrices into 100,089-dimensional vectors
3. **Apply PCA**: Reduce to 5 principal components
   - PC1 captures **98.64%** of variance (dominant 1D manifold structure)
   - Cumulative variance with 5 PCs: **99.95%**
4. **KNN prediction**: Use 10 nearest neighbors with inverse distance weighting
   - Find 10 nearest training trajectories in PCA space
   - Weight predictions by inverse distance: w_i = 1/(d_i + �)
   - Predict validation PCA components as weighted average
5. **Reconstruct**: Inverse PCA transform back to full trajectory space
6. **Evaluate**: Calculate errors at multiple granularities

### Results

#### Overall Performance

**Mean Absolute Percentage Error (MAPE):**
- **Mean MAPE**: 0.057%
- **Median MAPE**: 0.021%
- **Max MAPE**: 1.97%

These results indicate **excellent average performance** across all trajectories and features.

#### Max Element Error Analysis

To identify worst-case scenarios, we calculated the maximum error for any single element (specific species at specific timestep) within each trajectory:

- **Mean Max Element Error**: 180.97%
- **Median Max Element Error**: 9.12%
- **Max Element Error**: 50,960.40%

**Key Insight**: While average errors are extremely low (<0.06%), individual elements can have very high errors. This suggests that:
1. Most predictions are highly accurate
2. Occasional extreme outliers exist (likely rare species with low abundances)
3. The error distribution is highly skewed

#### Per-Species Error Breakdown

Calculated MAPE for each of the 333 species, averaged over all timesteps and validation trajectories:

**Top 20 Worst-Performing Species:**

| Rank | Species | MAPE (%) | Notes |
|------|---------|----------|-------|
| 1 | S | 0.21% | Atomic sulfur |
| 2 | C2H2 | 0.18% | Acetylene |
| 3 | #C | 0.18% | Surface carbon |
| 4 | C2H4 | 0.18% | Ethylene |
| 5 | CO2 | 0.17% | Carbon dioxide |
| 6 | HS2 | 0.17% | Disulfane radical |
| 7 | @HE | 0.16% | Bulk ice helium |
| 8 | S+ | 0.16% | Sulfur ion |
| 9 | H2S2+ | 0.16% | Protonated disulfane |
| 10 | #HCO | 0.16% | Surface formyl |
| 11 | C3H4 | 0.15% | Propyne/allene |
| 12 | C2H2O | 0.15% | Ketene |
| 13 | SIH4 | 0.15% | Silane |
| 14 | SIH3 | 0.15% | Silyl radical |
| 15 | S2+ | 0.15% | Disulfur cation |
| 16 | OCS | 0.14% | Carbonyl sulfide |
| 17 | SO+ | 0.14% | Sulfur monoxide ion |
| 18 | C2H3 | 0.14% | Vinyl radical |
| 19 | O | 0.13% | Atomic oxygen |
| 20 | SIO | 0.13% | Silicon monoxide |

**Error Distribution:**
- **Species with MAPE > 50%**: 0
- **Species with MAPE > 100%**: 0
- **All species errors < 0.21%**

Even the "worst" species have remarkably low errors, demonstrating that the manifold structure is well-preserved across all chemical species.

### Visualizations Generated

Created 7 comprehensive plots in `outputs/plots/knn_analysis/`:

1. **pca_variance.png**: Individual and cumulative explained variance
2. **pca_space_2d.png**: PC1 vs PC2 colored by density and error
3. **knn_distance_analysis.png**: Error vs KNN distance correlations + distributions
4. **neighbor_examples.png**: Example validation points with their nearest neighbors
5. **tsne_manifold_errors.png**: t-SNE visualization colored by MAPE
6. **tsne_manifold_max_errors.png**: t-SNE colored by max element error (log scale)
7. **species_error_analysis.png**: 4-panel species-level breakdown
   - Species error distribution histogram
   - Top 20 worst species bar chart
   - Cumulative error contribution
   - Summary statistics table

### Key Findings

1. **Strong 1D Manifold Structure**: PC1 alone captures 98.64% of variance, confirming that chemical trajectories lie on a low-dimensional manifold strongly correlated with physical evolution (likely dominated by density evolution).

2. **KNN Works Remarkably Well**: Simple weighted averaging in 5D PCA space achieves <0.06% mean error across 333 species and 297 timesteps. This suggests:
   - The manifold is smooth and continuous
   - Similar physical trajectories lead to similar chemistry
   - Local interpolation is valid in this space

3. **Sulfur Chemistry Most Challenging**: Sulfur-bearing species (S, HS2, S+, S2+, OCS, SO+) dominate the top error list, though still with <0.21% MAPE. This may indicate:
   - More complex/nonlinear chemistry
   - Less populated regions of parameter space
   - Possible grain surface chemistry complications

4. **Extreme Outliers Exist**: While median max element error is only 9.12%, the maximum reaches 50,960%. These likely represent:
   - Trace species with extremely low abundances (near 10{�p floor)
   - Edge cases in parameter space poorly represented in training
   - Possible timesteps where rapid chemical transitions occur

5. **PCA Compression is Highly Effective**: Reducing 100,089 dimensions to 5 loses only 0.05% of variance, enabling:
   - Efficient nearest-neighbor search
   - Computational speedup
   - Potential for further analysis and model development

### Implications

**For Neural Network Training:**
- The strong manifold structure validates the autoencoder + emulator approach
- PCA baseline (0.057% MAPE) sets a high bar for neural surrogates to beat
- Per-species error analysis can guide architecture improvements (e.g., special handling for sulfur chemistry)

**For Physical Understanding:**
- 1D manifold structure suggests chemistry is primarily driven by single parameter evolution (likely density)
- Smooth interpolation success implies predictable chemical evolution pathways
- Sulfur chemistry complexity may require special attention in full physics models

**Next Steps:**
1. Compare neural network emulator performance to this KNN baseline
2. Investigate specific trajectories with highest max element errors
3. Analyze temporal evolution of errors (which timesteps are hardest?)
4. Explore non-uniform weighting schemes for rare species
5. Test whether neural networks can eliminate the extreme outliers while maintaining low average error

### Code Implementation

Enhanced `scripts/datasets/plot-knn-analysis.py` with:
- Per-species error calculation and ranking
- Max element error analysis
- Log-scaled visualizations for handling extreme outliers
- Comprehensive console reporting
- 7 visualization outputs covering multiple error perspectives

All analysis reproducible by running:
```bash
python scripts/datasets/plot-knn-analysis.py
```
