# Complicated Typing Cases

This document describes complicated typing cases encountered during the codebase typing improvements.

## 1. Dataclass Configuration Classes Used as Both Types and Instances

**Files affected:** `src/training/train_emulator.py`, `src/training/train_autoencoder.py`

**Issue:** The configuration classes (`GeneralConfig`, `AEConfig`, `EMConfig`) are defined as dataclasses but are used as class-level containers of constants rather than instantiated objects. Functions expect instances but receive the class type itself.

**Example:**
```python
# Configuration is used directly as a class, not instantiated
processing_functions = dp.Processing(GeneralConfig, AEConfig)
# But the function signature expects an instance:
def __init__(self, GeneralConfig: GeneralConfig, ...) -> None:
```

**Resolution:** This is a design pattern choice. The type annotations are correct, but the actual usage passes the class instead of an instance. This works at runtime because Python allows accessing class attributes on both the class and instances. Consider either:
- Instantiating the config classes: `GeneralConfig()`
- Or changing type hints to `type[GeneralConfig]` if classes are meant to be used directly

## 2. Optional Model Parameters Without Null Checks

**File affected:** `src/inference.py`

**Issue:** The `Inference` class accepts optional `autoencoder` and `emulator` parameters, but methods like `encode()` and `decode()` call methods on them without checking for `None`.

**Example:**
```python
def __init__(
    self,
    ...
    autoencoder: Optional[Autoencoder] = None,
    emulator: Optional[Emulator] = None,
) -> None:
    ...

def encode(self, abundances) -> torch.Tensor:
    # No null check - will fail if autoencoder is None
    latents = self.autoencoder.encode(abundances)
```

**Resolution:** Add runtime checks or use `assert` statements before using optional parameters.

## 3. Undefined `DatasetConfig` Reference

**File affected:** `src/analysis.py`

**Issue:** The code references `DatasetConfig` which is not defined anywhere in the codebase. This appears to be a legacy reference that should be `GeneralConfig`.

**Resolution:** Replace all `DatasetConfig` references with `GeneralConfig`.

## 4. Missing `PredefinedTensors` Import

**File affected:** `src/analysis.py`

**Issue:** The code tries to import `PredefinedTensors` from `configs.general` but this class doesn't exist in that module.

**Resolution:** Either define `PredefinedTensors` in `configs/general.py` or remove the import and define the constants locally.

## 5. Generator Function Return Type

**File affected:** `src/data_loading.py`

**Issue:** The `__iter__` method in `ChunkedShuffleSampler` uses `yield from`, making it a generator function. Adding a return type annotation of `-> None` causes type errors.

**Example:**
```python
def __iter__(self):  # Cannot add -> None due to generator semantics
    yield from chunk_perm.tolist()
```

**Resolution:** Leave the return type annotation off for generator methods that use `yield`.

## 6. DistributedSampler-Specific Methods

**File affected:** `src/trainer.py`

**Issue:** The code calls `sampler.set_epoch(epoch)` on DataLoader samplers, but this method only exists on `DistributedSampler`, not the base `Sampler` class.

**Example:**
```python
self.training_dataloader.sampler.set_epoch(epoch)
# Error: Cannot access attribute "set_epoch" for class "Sampler[Unknown]"
```

**Resolution:** Either:
- Type the sampler specifically as `DistributedSampler`
- Use `# type: ignore` comments
- Add a runtime check: `if hasattr(self.training_dataloader.sampler, 'set_epoch')`

## 7. torch.jit.script Function Restrictions

**File affected:** `src/analysis.py`, `src/data_processing.py`

**Issue:** Functions decorated with `@torch.jit.script` have restrictions on type annotations and cannot use certain Python features like complex union types or external module references.

**Example:**
```python
@torch.jit.script
def calculate_conservation_error(tensor1: torch.Tensor, tensor2: torch.Tensor):
    # Cannot reference external module functions directly
    unscaled_tensor1 = dp.stoichiometric_matrix_mult(tensor1)  # Error
```

**Resolution:** JIT-compiled functions should be self-contained or receive all necessary data as parameters.

## 8. DataFrame Column Type Inference

**File affected:** `src/preprocessing/preprocessing.py`

**Issue:** Pandas `Index` type from `df.columns` doesn't match `List[str]` type hint in function parameter.

**Example:**
```python
df.columns = utils.rename_columns(df.columns)
# rename_columns expects List[str] but df.columns is Index[str]
```

**Resolution:** Convert to list when needed: `list(df.columns)` or update type hints to accept `List[str] | pd.Index`.
