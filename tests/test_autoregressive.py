"""Basic tests for abundance autoregressive additions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.configs.autoregressive import AutoregressiveConfig
from src.configs.datasets import DatasetConfig
from src.data_loading import AutoregressiveSequenceDataset
from src.data_processing import Processing, preprocessing_autoregressive_dataset
from src.models.autoregressive import Autoregressive


def _write_json(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


class AutoregressiveTests(unittest.TestCase):
    """Coverage for normal-space autoregressive helpers."""

    def setUp(self) -> None:
        """Create an isolated dataset fixture on disk."""
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        preprocess_dir = root / "outputs" / "preprocessed" / "toy"
        preprocess_dir.mkdir(parents=True, exist_ok=True)

        _write_json(
            preprocess_dir / "physical_parameter_ranges.json",
            '{"density": {"min": 1.0, "max": 100.0}, "temp": {"min": 10.0, "max": 1000.0}}',
        )
        _write_json(preprocess_dir / "species.json", '{"species": ["A", "B"]}')
        _write_json(
            preprocess_dir / "columns.json",
            '{"0": "Index", "1": "Model", "2": "Time", "3": "density", "4": "temp", "5": "A", "6": "B"}',
        )
        np.save(preprocess_dir / "stoichiometric_matrix.npy", np.eye(2, dtype=np.float32))

        self.project_root = root

    def tearDown(self) -> None:
        """Clean up the temporary dataset fixture."""
        self.tempdir.cleanup()

    def _dataset_config(self) -> DatasetConfig:
        return DatasetConfig(
            dataset_name="toy",
            working_path=str(self.project_root),
            project_root=str(self.project_root),
            device=torch.device("cpu"),
        )

    def test_autoregressive_config_derives_expected_dims(self) -> None:
        """Autoregressive config should derive input/output dimensions from dataset metadata."""
        dataset_config = self._dataset_config()
        config = AutoregressiveConfig(dataset_config=dataset_config)

        self.assertEqual(config.input_dim, 4)
        self.assertEqual(config.output_dim, 2)
        self.assertTrue(config.save_model_path.endswith("autoregressive.pth"))

    def test_preprocessing_and_sequence_dataset_shapes(self) -> None:
        """Preprocessing and sequence slicing should expose the expected tensor shapes."""
        dataset_config = self._dataset_config()
        ar_config = AutoregressiveConfig(dataset_config=dataset_config, window_size=3)
        processing = Processing(dataset_config)
        dataset_np = np.array(
            [
                [0.0, 7.0, 0.0, 1.0, 10.0, 1e-4, 1e-5],
                [1.0, 7.0, 1.0, 10.0, 100.0, 1e-3, 1e-4],
                [2.0, 7.0, 2.0, 100.0, 1000.0, 1e-2, 1e-3],
                [3.0, 7.0, 3.0, 100.0, 1000.0, 1e-1, 1e-2],
            ],
            dtype=np.float32,
        )

        data_matrix, index_pairs = preprocessing_autoregressive_dataset(
            dataset_config,
            ar_config,
            dataset_np,
            processing,
        )
        sequence_dataset = AutoregressiveSequenceDataset(
            dataset_config,
            data_matrix,
            index_pairs,
        )

        phys, features, targets = sequence_dataset.__getitems__([0])

        self.assertEqual(tuple(phys.shape), (1, 2, 2))
        self.assertEqual(tuple(features.shape), (1, 2))
        self.assertEqual(tuple(targets.shape), (1, 2, 2))

    def test_autoregressive_forward_shape(self) -> None:
        """Forward pass should preserve batch/time layout and predict per-species outputs."""
        model = Autoregressive(input_dim=4, output_dim=2, hidden_dim=8)
        phys = torch.rand(3, 5, 2)
        abundances = torch.rand(3, 2)
        outputs = model(phys, abundances)

        self.assertEqual(tuple(outputs.shape), (3, 5, 2))


if __name__ == "__main__":
    unittest.main()
