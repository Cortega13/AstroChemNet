"""Autoregressive training entrypoint."""

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
from src.models.autoregressive.config import AutoregressiveConfig, build_config
from src.models.autoregressive.data import (
    AutoregressiveSequenceDataset,
    ensure_preprocessed,
)
from src.models.autoregressive.model import Autoregressive, load_autoregressive
from src.training import (
    TORCH_COMPILE_MODE,
    Trainer,
    load_objects,
)


class AutoregressiveTrainerSequential(Trainer):
    """Trainer for abundance autoregressive models using sequential processing."""

    def __init__(
        self,
        general_config: DatasetConfig,
        ar_config: AutoregressiveConfig,
        loss_functions: Loss,
        autoregressive: Autoregressive,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """Initialize trainer for abundance autoregressive model."""
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation
        self.gradient_clipping = ar_config.gradient_clipping

        super().__init__(
            general_config,
            model_config=ar_config,
            model=autoregressive,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )

        self.model = cast(
            Autoregressive,
            torch.compile(self.model, mode=TORCH_COMPILE_MODE),
        )

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
            phys, features, targets = next(iterator)
            self._run_training_batch(
                phys.to(self.device, non_blocking=True),
                features.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
            )

        activities = [torch.profiler.ProfilerActivity.CPU] + (
            [torch.profiler.ProfilerActivity.CUDA] if str(self.device) == "cuda" else []
        )
        with torch.profiler.profile(activities=activities) as profiler:
            for _ in range(prof_batches):
                phys, features, targets = next(iterator)
                self._run_training_batch(
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
            trace_path = self._profile_trace_path("autoregressive")
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            profiler.export_chrome_trace(trace_path)

    def _run_training_batch(
        self,
        phys: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Run a single training batch for the abundance autoregressive."""
        self.optimizer.zero_grad()
        outputs = self.model(phys, features).reshape(-1, features.shape[-1])
        loss = self.training_loss(outputs, targets.reshape(-1, targets.shape[-1]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(
        self,
        phys: torch.Tensor,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Run a single validation batch for the abundance autoregressive."""
        outputs = self.model(phys, features)
        self.epoch_validation_loss += self.validation_loss(outputs, targets).mean(dim=0).detach()

    def _run_epoch(self, epoch: int) -> None:
        """Run a single epoch of training and validation for the autoregressive."""
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
    """Train abundance autoregressive model with given configuration."""
    ensure_preprocessed(dataset_config, force=force_preprocess)
    ar_config = build_config(dataset_config)
    training_tensors = load_tensors(
        dataset_config,
        category="training_seq",
        artifact_dir="autoregressive",
    )
    validation_tensors = load_tensors(
        dataset_config,
        category="validation_seq",
        artifact_dir="autoregressive",
    )
    training_dataset = training_tensors["dataset"]
    training_indices = training_tensors["indices"]
    validation_dataset = validation_tensors["dataset"]
    validation_indices = validation_tensors["indices"]

    training_sequence_dataset = AutoregressiveSequenceDataset(
        dataset_config,
        training_dataset,
        training_indices,
    )
    validation_sequence_dataset = AutoregressiveSequenceDataset(
        dataset_config,
        validation_dataset,
        validation_indices,
    )

    processing_functions = dp.Processing(dataset_config)
    loss_functions = Loss(
        processing_functions,
        dataset_config,
        ModelConfig=ar_config,
    )
    training_dataloader = tensor_to_dataloader(ar_config, training_sequence_dataset)
    validation_dataloader = tensor_to_dataloader(
        ar_config,
        validation_sequence_dataset,
    )
    autoregressive = load_autoregressive(Autoregressive, dataset_config, ar_config)
    optimizer, scheduler = load_objects(autoregressive, ar_config)
    trainer = AutoregressiveTrainerSequential(
        dataset_config,
        ar_config,
        loss_functions,
        autoregressive,
        optimizer,
        scheduler,
        training_dataloader,
        validation_dataloader,
    )
    trainer.train()
