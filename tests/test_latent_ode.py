"""Tests for latent ODE config, preprocessing, and model rollout."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.configs.autoencoder import AEConfig
from src.configs.datasets import DatasetConfig
from src.configs.latent_ode import LatentODEConfig
from src.data_loading import LatentODESequenceDataset
from src.data_processing import (
    Processing,
    compute_base_dt,
    preprocessing_latent_ode_dataset,
)
from src.inference import Inference
from src.models.autoencoder import Autoencoder
from src.models.latent_ode import LatentODE


def _write_json(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


class LatentODETests(unittest.TestCase):
    """Coverage for latent ODE additions."""

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
        np.save(preprocess_dir / "latents_minmax.npy", np.array([0.0, 1.0], dtype=np.float32))

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

    def test_latent_ode_config_derives_expected_dims(self) -> None:
        """Latent ODE config should derive dimensions and paths from dataset metadata."""
        dataset_config = self._dataset_config()
        ae_config = AEConfig(dataset_config=dataset_config, latent_dim=3, hidden_dims=(8, 4))
        ode_config = LatentODEConfig(dataset_config=dataset_config, ae_config=ae_config)

        self.assertEqual(ode_config.input_dim, 5)
        self.assertEqual(ode_config.output_dim, 3)
        self.assertTrue(ode_config.save_model_path.endswith("latent_ode.pth"))
        self.assertTrue(ode_config.base_dt_path.endswith("latent_ode/base_dt.json"))

    def test_preprocessing_and_sequence_dataset_shapes(self) -> None:
        """Preprocessing should produce normalized deltas and expected tensor shapes."""
        dataset_config = self._dataset_config()
        ae_config = AEConfig(dataset_config=dataset_config, latent_dim=3, hidden_dims=(8, 4))
        ode_config = LatentODEConfig(
            dataset_config=dataset_config,
            ae_config=ae_config,
            window_size=3,
        )
        autoencoder = Autoencoder(
            input_dim=ae_config.input_dim,
            latent_dim=ae_config.latent_dim,
            hidden_dims=ae_config.hidden_dims,
            noise=0.0,
            dropout=0.0,
        )
        processing = Processing(dataset_config, ae_config)
        inference = Inference(dataset_config, processing, autoencoder)
        dataset_np = np.array(
            [
                [0.0, 7.0, 0.0, 1.0, 10.0, 1e-4, 1e-5],
                [1.0, 7.0, 5.0, 10.0, 100.0, 1e-3, 1e-4],
                [2.0, 7.0, 10.0, 100.0, 1000.0, 1e-2, 1e-3],
                [3.0, 7.0, 15.0, 100.0, 1000.0, 1e-1, 1e-2],
            ],
            dtype=np.float32,
        )

        base_dt = compute_base_dt(dataset_np)
        data_matrix, index_pairs, delta_t = preprocessing_latent_ode_dataset(
            dataset_config,
            ode_config,
            dataset_np,
            processing,
            inference,
            base_dt,
        )
        sequence_dataset = LatentODESequenceDataset(
            dataset_config,
            data_matrix,
            index_pairs,
            delta_t,
            ae_config.latent_dim,
        )

        step_dt, phys, features, targets = sequence_dataset.__getitems__([0])

        self.assertEqual(base_dt, 5.0)
        self.assertEqual(tuple(step_dt.shape), (1, 2))
        self.assertTrue(torch.allclose(step_dt, torch.ones_like(step_dt)))
        self.assertEqual(tuple(phys.shape), (1, 2, 2))
        self.assertEqual(tuple(features.shape), (1, 3))
        self.assertEqual(tuple(targets.shape), (1, 2, 2))

    def test_latent_ode_zero_dynamics_preserves_state(self) -> None:
        """Zero dynamics should keep the latent state constant over rollout."""
        model = LatentODE(
            latent_dim=3,
            phys_dim=2,
            hidden_dim=8,
            num_hidden_layers=1,
            method="rk4",
            solver_substeps=2,
        )
        for parameter in model.parameters():
            parameter.data.zero_()

        delta_t = torch.ones(2, 4)
        phys = torch.rand(2, 4, 2)
        latents = torch.rand(2, 3)
        outputs = model(delta_t, phys, latents)

        expected = latents.unsqueeze(1).expand_as(outputs)
        self.assertEqual(tuple(outputs.shape), (2, 4, 3))
        self.assertTrue(torch.allclose(outputs, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
