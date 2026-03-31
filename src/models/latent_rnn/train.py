"""Latent RNN training entrypoint."""

from datetime import datetime
from typing import cast

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src import data_processing as dp
from src.data_loading import load_tensors, tensor_to_dataloader
from src.datasets import DatasetConfig
from src.loss import Loss
from src.models.autoencoder.config import AEConfig
from src.models.autoencoder.config import build_config as build_ae_config
from src.models.autoencoder.model import Autoencoder, load_autoencoder
from src.models.latent_rnn.config import LatentRNNConfig, build_config
from src.models.latent_rnn.data import LatentSequenceDataset, ensure_preprocessed
from src.models.latent_rnn.model import LatentRNN, load_latent_rnn
from src.training import TORCH_COMPILE_MODE, Trainer, load_objects


class LatentRNNTrainerSequential(Trainer):
    """Trainer for latent RNN models using sequential processing."""

    def __init__(
        self,
        general_config: DatasetConfig,
        ae_config: AEConfig,
        model_config: LatentRNNConfig,
        loss_functions: Loss,
        processing_functions: dp.Processing,
        autoencoder: Autoencoder,
        latent_rnn: LatentRNN,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """Initialize latent RNN trainer."""
        self.ae = autoencoder
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation
        self.latent_dim = ae_config.latent_dim
        self.num_species = general_config.num_species
        self.gradient_clipping = model_config.gradient_clipping
        self.inverse_latent_components_scaling = (
            processing_functions.inverse_latent_components_scaling
        )

        super().__init__(
            general_config,
            model_config=model_config,
            model=latent_rnn,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )

        self.model = cast(LatentRNN, torch.compile(self.model, mode=TORCH_COMPILE_MODE))

    def _run_training_batch(self, phys: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Run a single training batch for the latent RNN."""
        self.optimizer.zero_grad()
        outputs = self.model(phys, features).reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
        loss = self.training_loss(outputs, targets.reshape(-1, self.num_species))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(self, phys: torch.Tensor, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Run a single validation batch for the latent RNN."""
        outputs = self.model(phys, features).reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs).reshape(targets.size(0), targets.size(1), -1)
        self.epoch_validation_loss += self.validation_loss(outputs, targets).mean(dim=0).detach()

    def _run_epoch(self, epoch: int) -> None:
        """Run a single epoch of training and validation."""
        sampler = cast(DistributedSampler, self.training_dataloader.sampler)
        sampler.set_epoch(epoch)
        tic1 = datetime.now()

        self.model.train()
        for phys, features, targets in self.training_dataloader:
            self._run_training_batch(
                phys.to(self.device, non_blocking=True),
                features.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
            )

        tic2 = datetime.now()

        self.model.eval()
        with torch.no_grad():
            for phys, features, targets in self.validation_dataloader:
                self._run_validation_batch(
                    phys.to(self.device, non_blocking=True),
                    features.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}")


def train(dataset_config, force_preprocess: bool = False) -> None:
    """Train latent RNN with cached latent sequence artifacts."""
    ensure_preprocessed(dataset_config, force=force_preprocess)
    ae_config = build_ae_config(dataset_config)
    model_config = build_config(dataset_config, ae_config)
    training_tensors = load_tensors(
        dataset_config,
        category="training_seq",
        artifact_dir="latent_rnn",
    )
    validation_tensors = load_tensors(
        dataset_config,
        category="validation_seq",
        artifact_dir="latent_rnn",
    )
    training_dataset = training_tensors["dataset"]
    training_indices = training_tensors["indices"]
    validation_dataset = validation_tensors["dataset"]
    validation_indices = validation_tensors["indices"]

    training_sequence_dataset = LatentSequenceDataset(
        dataset_config,
        training_dataset,
        training_indices,
        ae_config.latent_dim,
    )
    validation_sequence_dataset = LatentSequenceDataset(
        dataset_config,
        validation_dataset,
        validation_indices,
        ae_config.latent_dim,
    )

    processing_functions = dp.Processing(dataset_config, ae_config)
    autoencoder = load_autoencoder(
        Autoencoder,
        dataset_config,
        ae_config,
        inference=True,
    )
    loss_functions = Loss(
        processing_functions,
        dataset_config,
        ModelConfig=model_config,
    )
    training_dataloader = tensor_to_dataloader(
        model_config,
        training_sequence_dataset,
    )
    validation_dataloader = tensor_to_dataloader(
        model_config,
        validation_sequence_dataset,
    )
    latent_rnn = load_latent_rnn(LatentRNN, dataset_config, model_config)
    optimizer, scheduler = load_objects(latent_rnn, model_config)
    trainer = LatentRNNTrainerSequential(
        dataset_config,
        ae_config,
        model_config,
        loss_functions,
        processing_functions,
        autoencoder,
        latent_rnn,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    trainer.train()
