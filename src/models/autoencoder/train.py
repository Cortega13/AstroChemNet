"""Autoencoder training entrypoint."""

from datetime import datetime
from typing import cast

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src import data_processing as dp
from src import ensure_dataset_preprocessed
from src.data_loading import load_datasets, tensor_to_dataloader
from src.datasets import DatasetConfig
from src.loss import Loss
from src.models.autoencoder.config import AEConfig, build_config
from src.models.autoencoder.data import AutoencoderDataset
from src.models.autoencoder.inference import Inference
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.training import TORCH_COMPILE_MODE, Trainer, load_objects


class AutoencoderTrainer(Trainer):
    """Trainer specialized for autoencoder models."""

    def __init__(
        self,
        general_config: DatasetConfig,
        ae_config: AEConfig,
        loss_functions: Loss,
        autoencoder: Autoencoder,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """Initialize autoencoder trainer."""
        self.gradient_clipping = ae_config.gradient_clipping
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation

        super().__init__(
            general_config,
            model_config=ae_config,
            model=autoencoder,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )

        self.model = cast(
            Autoencoder,
            torch.compile(self.model, mode=TORCH_COMPILE_MODE),
        )

    def _run_training_batch(self, features: torch.Tensor) -> None:
        """Run a training batch where features equal targets."""
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.training_loss(outputs, features)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(self, features: torch.Tensor) -> None:
        """Run a validation batch where features equal targets."""
        outputs = self.model.decode(self.model.encode(features))
        self.epoch_validation_loss += self.validation_loss(outputs, features)

    def _run_epoch(self, epoch: int) -> None:
        """Run one autoencoder training and validation epoch."""
        sampler = cast(DistributedSampler, self.training_dataloader.sampler)
        sampler.set_epoch(epoch)

        tic1 = datetime.now()
        self.model.train()
        for features in self.training_dataloader:
            self._run_training_batch(features[0].to(self.device, non_blocking=True))

        tic2 = datetime.now()
        self.model.eval()
        with torch.no_grad():
            for features in self.validation_dataloader:
                self._run_validation_batch(
                    features[0].to(self.device, non_blocking=True)
                )

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}\n")


def save_latents_minmax(
    general_config,
    ae_config,
    dataset_t: torch.Tensor,
    inference_functions: Inference,
) -> None:
    """Compute and save min/max values of latent components for scaling."""
    min_, max_ = float("inf"), float("-inf")

    with torch.no_grad():
        for i in range(0, len(dataset_t), ae_config.batch_size):
            batch = dataset_t[i : i + ae_config.batch_size].to(general_config.device)
            encoded = inference_functions.encode(batch).cpu()
            min_ = min(min_, encoded.min().item())
            max_ = max(max_, encoded.max().item())

    minmax_np = np.array([min_, max_], dtype=np.float32)
    print(f"Latents MinMax: {minmax_np[0]}, {minmax_np[1]}")
    np.save(ae_config.latents_minmax_path, minmax_np)


def train(dataset_config, force_preprocess: bool = False) -> None:
    """Train autoencoder model with given configuration."""
    ensure_dataset_preprocessed(dataset_config.name, force=force_preprocess)
    ae_config = build_config(dataset_config)
    processing_functions = dp.Processing(dataset_config)

    training_np, validation_np = load_datasets(dataset_config, ae_config.columns)

    processing_functions.abundances_scaling(training_np)
    processing_functions.abundances_scaling(validation_np)
    training_dataset = torch.from_numpy(training_np)
    validation_dataset = torch.from_numpy(validation_np)

    training_dataset_wrapped = AutoencoderDataset(training_dataset)
    validation_dataset_wrapped = AutoencoderDataset(validation_dataset)

    loss_functions = Loss(
        processing_functions,
        dataset_config,
        ModelConfig=ae_config,
    )
    autoencoder = load_autoencoder(Autoencoder, dataset_config, ae_config)
    training_dataloader = tensor_to_dataloader(ae_config, training_dataset_wrapped)
    validation_dataloader = tensor_to_dataloader(
        ae_config,
        validation_dataset_wrapped,
    )
    optimizer, scheduler = load_objects(autoencoder, ae_config)
    trainer = AutoencoderTrainer(
        dataset_config,
        ae_config,
        loss_functions,
        autoencoder,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    trainer.train()

    total_dataset = torch.vstack((training_dataset, validation_dataset))
    inference_functions = Inference(dataset_config, processing_functions, autoencoder)
    save_latents_minmax(dataset_config, ae_config, total_dataset, inference_functions)
