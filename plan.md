# Memory Optimization Plan for `scripts/preprocessing.py`

## Status: ✅ COMPLETED

The implementation has been successfully completed and tested.

## Original Memory Issues

The original script had several memory bottlenecks:

1. **Line 25**: Loads entire HDF5 file into memory at once
2. **Lines 41-53**: Accumulates all processed chunks in `output_chunks` list before concatenation
3. **Line 55**: `pd.concat(output_chunks)` creates another copy in memory
4. **Line 64**: `df.to_numpy()` creates yet another copy

## Implemented Solution: Batch Processing with Intermediate HDF5

### Key Changes

1. **Batch Processing**: Process models in batches of 100 models at a time
2. **Intermediate HDF5**: Write each batch to an intermediate HDF5 file
3. **Memory Cleanup**: Explicitly delete intermediate variables after each batch
4. **Final Conversion**: Read intermediate file and convert to final `.npy` format

### Results

- **Input**: 2,953,784 rows, 9,979 unique models
- **Output**: 2,963,763 rows × 340 columns (final array shape)
- **Processing**: 100 batches of ~100 models each
- **Memory**: Successfully completed without OOM errors

### Implementation Details

The final implementation in [`scripts/preprocessing.py`](scripts/preprocessing.py):

1. Loads initial abundances (small, stays in memory)
2. Reads a small sample to determine species columns
3. Loads the full source data (necessary since HDF5 is not in table format)
4. Processes models in batches of 100:
   - Filters to batch models
   - Processes each model (adds initial abundances, shifts physical params)
   - Writes batch to intermediate HDF5
   - Cleans up batch variables
5. Reads intermediate HDF5 and converts to final `.npy` format
6. Cleans up intermediate file

### Memory Estimates

| Stage | Memory Usage |
|-------|-------------|
| Initial abundances | ~1MB |
| Source HDF5 read | ~7GB |
| Batch processing (per batch) | ~100-200MB |
| Intermediate HDF5 write | Minimal (streaming) |
| Final conversion | ~7GB |
| **Peak** | ~8-10GB |

This keeps peak memory well under the 32GB limit and within the 2-3x dataset size constraint.
