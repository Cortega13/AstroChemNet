"""Latent ODE training entrypoint."""

import os
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
from src.models.latent_ode.config import LatentODEConfig, build_config
from src.models.latent_ode.data import LatentODESequenceDataset, ensure_preprocessed
from src.models.latent_ode.model import LatentODE, load_latent_ode
from src.training import (
    Trainer,
    load_objects,
)


class LatentODETrainerSequential(Trainer):
    """Trainer for latent ODE models using full-window rollout."""

    def __init__(
        self,
        general_config: DatasetConfig,
        ae_config: AEConfig,
        ode_config: LatentODEConfig,
        loss_functions: Loss,
        processing_functions: dp.Processing,
        autoencoder: Autoencoder,
        latent_ode: LatentODE,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """Initialize trainer for latent ODE model."""
        self.ae = autoencoder
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation
        self.latent_dim = ae_config.latent_dim
        self.num_species = general_config.num_species
        self.gradient_clipping = ode_config.gradient_clipping
        self.inverse_latent_components_scaling = (
            processing_functions.inverse_latent_components_scaling
        )

        super().__init__(
            general_config,
            model_config=ode_config,
            model=latent_ode,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )

        self.model = cast(LatentODE, torch.compile(self.model))

    def _profile_epoch(self, warmup_batches: int = 3, prof_batches: int = 1) -> None:
        """Profile a short training run and save a chrome trace."""
        self.model.train()
        warmup_batches, prof_batches = self._resolve_profile_batch_counts(
            warmup_batches,
            prof_batches,
        )
        if prof_batches == 0:
            return
        iterator = iter(self.training_dataloader)
        for _ in range(warmup_batches):
            delta_t, phys, features, targets = next(iterator)
            self._run_training_batch(
                delta_t.to(self.device, non_blocking=True),
                phys.to(self.device, non_blocking=True),
                features.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
            )

        activities = [torch.profiler.ProfilerActivity.CPU] + (
            [torch.profiler.ProfilerActivity.CUDA] if str(self.device) == "cuda" else []
        )
        with torch.profiler.profile(activities=activities) as profiler:
            for _ in range(prof_batches):
                delta_t, phys, features, targets = next(iterator)
                self._run_training_batch(
                    delta_t.to(self.device, non_blocking=True),
                    phys.to(self.device, non_blocking=True),
                    features.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )
                profiler.step()

        if str(self.device) == "cuda":
            torch.cuda.synchronize()
        if not torch.distributed.is_available() or (
            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        ):
            trace_path = self._profile_trace_path("latent_ode")
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            profiler.export_chrome_trace(trace_path)

    def _run_training_batch(
        self,
        delta_t: torch.Tensor,
        phys: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Run one training batch for the latent ODE model."""
        self.optimizer.zero_grad()
        outputs = self.model(delta_t, phys, features).reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
        loss = self.training_loss(outputs, targets.reshape(-1, self.num_species))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(
        self,
        delta_t: torch.Tensor,
        phys: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Run one validation batch for the latent ODE model."""
        outputs = self.model(delta_t, phys, features).reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs).reshape(targets.size(0), targets.size(1), -1)
        self.epoch_validation_loss += (
            self.validation_loss(outputs, targets).mean(dim=0).detach()
        )

    def _run_epoch(self, epoch: int) -> None:
        """Run a full training and validation epoch for latent ODE."""
        sampler = cast(DistributedSampler, self.training_dataloader.sampler)
        sampler.set_epoch(epoch)
        tic1 = datetime.now()

        self.model.train()
        for delta_t, phys, features, targets in self.training_dataloader:
            self._run_training_batch(
                delta_t.to(self.device, non_blocking=True),
                phys.to(self.device, non_blocking=True),
                features.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
            )

        tic2 = datetime.now()

        self.model.eval()
        with torch.no_grad():
            for delta_t, phys, features, targets in self.validation_dataloader:
                self._run_validation_batch(
                    delta_t.to(self.device, non_blocking=True),
                    phys.to(self.device, non_blocking=True),
                    features.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}")


def train(dataset_config, force_preprocess: bool = False) -> None:
    """Train latent ODE with cached interval-aware latent artifacts."""
    ensure_preprocessed(dataset_config, force=force_preprocess)
    ae_config = build_ae_config(dataset_config)
    model_config = build_config(dataset_config, ae_config)
    training_tensors = load_tensors(
        dataset_config,
        category="training_seq",
        artifact_dir="latent_ode",
    )
    validation_tensors = load_tensors(
        dataset_config,
        category="validation_seq",
        artifact_dir="latent_ode",
    )

    training_dataset = LatentODESequenceDataset(
        dataset_config,
        training_tensors["dataset"],
        training_tensors["indices"],
        training_tensors["delta_t"],
        ae_config.latent_dim,
    )
    validation_dataset = LatentODESequenceDataset(
        dataset_config,
        validation_tensors["dataset"],
        validation_tensors["indices"],
        validation_tensors["delta_t"],
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
    training_dataloader = tensor_to_dataloader(model_config, training_dataset)
    validation_dataloader = tensor_to_dataloader(model_config, validation_dataset)
    latent_ode = load_latent_ode(LatentODE, dataset_config, model_config)
    optimizer, scheduler = load_objects(latent_ode, model_config)
    trainer = LatentODETrainerSequential(
        dataset_config,
        ae_config,
        model_config,
        loss_functions,
        processing_functions,
        autoencoder,
        latent_ode,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    trainer.train()
