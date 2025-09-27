from abc import abstractmethod
import os
import gc
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.backends import cudnn
import copy
import json

cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(
        self,
        GeneralConfig,
        model_config,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """
        Initializes the Trainer class. A class which simplifies training by including all necessary components.
        """
        self.device = GeneralConfig.device
        self.start_time = datetime.now()
        self.model_config = model_config

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.num_validation_elements = len(self.validation_dataloader.dataset)

        self.current_dropout_rate = self.model_config.dropout
        self.current_learning_rate = self.model_config.lr
        self.best_weights = None
        self.metric_minimum_loss = np.inf
        self.epoch_validation_loss = torch.zeros(GeneralConfig.num_species).to(
            self.device
        )
        self.stagnant_epochs = 0
        self.loss_per_epoch = []

    def save_loss_per_epoch(self):
        """
        Saves the loss per epoch to a file.
        """
        epochs_path = os.path.splitext(self.model_config.save_model_path)[0] + ".json"
        with open(epochs_path, "w") as f:
            json.dump(self.loss_per_epoch, f, indent=4)

    def print_final_time(self):
        """
        Prints the total training time.
        """
        end_time = datetime.now()
        total_time = end_time - self.start_time
        print(f"Total Training Time: {total_time}")
        print(f"Total Epochs: {len(self.loss_per_epoch)}")

    def _save_checkpoint(self):
        """
        Saves the model's state dictionary to a file.
        """
        checkpoint = self.model.state_dict()
        model_path = os.path.join(self.model_config.save_model_path)
        if self.model_config.save_model:
            torch.save(checkpoint, model_path)

    def set_dropout_rate(self, dropout_rate):
        """
        Sets the dropout rate for all dropout layers in the model.
        """
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate
        self.current_dropout_rate = dropout_rate

    def _check_early_stopping(self):
        """
        Ends training once the number of stagnant epochs exceeds the patience.
        """
        if self.stagnant_epochs >= self.model_config.stagnant_epoch_patience:
            print("Ending training early due to stagnant epochs.")
            return True
        return False

    def _check_minimum_loss(self):
        """
        Checks if the current epoch's validation loss is the minimum loss so far.
        Calculates the mean relative error and the std of the species-wise mean relative error.
        Uses a metric for the minimum loss which gives weight to the mean and std relative errors.
        Includes a scheduler to reduce the learning rate once the minimum loss stagnates.
        """
        val_loss = self.epoch_validation_loss / self.num_validation_elements
        mean_loss = val_loss.mean().item()
        std_loss = val_loss.std().item()
        max_loss = val_loss.max().item()
        metric = mean_loss  # + std_loss + 0.5*max_loss

        if metric < self.metric_minimum_loss:
            print("**********************")
            print(
                f"New Minimum \nMean: {mean_loss:.3e} \nStd: {std_loss:.3e} \nMax: {max_loss:.3e} \nMetric: {metric:.3e} \nPercent Improvement: {(100 - metric * 100 / self.metric_minimum_loss):.3f}%"
            )
            self._save_checkpoint()
            self.best_weights = copy.deepcopy(self.model.state_dict())

            self.metric_minimum_loss = metric
            self.stagnant_epochs = 0
        else:
            self.stagnant_epochs += 1
            print(
                f"Stagnant {self.stagnant_epochs} \nMinimum: {self.metric_minimum_loss:.3e} \nMean: {mean_loss:.3e} \nStd: {std_loss:.3e} \nMax: {max_loss:.3e} \nMetric: {metric:.3e}"
            )

            if self.stagnant_epochs % self.model_config.dropout_decay_patience == 0:
                new_dropout = max(
                    self.current_dropout_rate
                    - self.model_config.dropout_reduction_factor,
                    0.0,
                )
                if new_dropout != self.current_dropout_rate:
                    self.stagnant_epochs = 0
                    self.set_dropout_rate(new_dropout)

                    self.current_learning_rate = (
                        1e-3 if new_dropout <= 0.1 else self.model_config.lr
                    )

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.current_learning_rate

                    print(
                        f"Decreasing dropout rate to {self.current_dropout_rate:.4f} and settings lr to {self.current_learning_rate:.4f}."
                    )

            if self.stagnant_epochs == self.model_config.lr_decay_patience + 1:
                print("Reverting to previous best weights")
                self.model.load_state_dict(self.best_weights)

        self.loss_per_epoch.append(
            {
                "mean": mean_loss,
                "std": std_loss,
                "max": max_loss,
                "metric": metric,
                "dropout": self.current_dropout_rate,
                "learning_rate": self.current_learning_rate,
            }
        )
        self.epoch_validation_loss.zero_()
        self.scheduler.step(metric)
        print()
        print(f"Current Learning Rate: {self.optimizer.param_groups[0]['lr']:.3e}")
        print(f"Current Dropout Rate: {self.current_dropout_rate:.4f}")
        print(f"Current Num Epochs: {len(self.loss_per_epoch)}")

    @abstractmethod
    def _run_epoch(self, epoch):
        return NotImplementedError("This method should be implemented in subclasses.")

    def train(self):
        """
        Training loop for the autoencoder. Runs until the minimum loss stagnates for a number of epochs.
        """
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        for epoch in range(9999999):
            self._run_epoch(epoch)
            self._check_minimum_loss()
            if self._check_early_stopping():
                break

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"\nTraining Complete. Trial Results: {self.metric_minimum_loss}")
        self.print_final_time()
        self.save_loss_per_epoch()


class AutoencoderTrainer(Trainer):
    def __init__(
        self,
        GeneralConfig,
        AEConfig,
        loss_functions,
        autoencoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """
        Initializes the AutoencoderTrainer, a subclass of Trainer, specialized for training the autoencoder.
        """
        self.num_metadata = GeneralConfig.num_metadata
        self.num_phys = GeneralConfig.phys
        self.num_species = GeneralConfig.num_species
        self.num_components = AEConfig.latent_dim
        self.gradient_clipping = AEConfig.gradient_clipping

        self.ae = autoencoder
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation

        super().__init__(
            GeneralConfig,
            model_config=AEConfig,
            model=autoencoder,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )

    def _run_training_batch(self, features):
        """
        Runs a training batch where features = targets since this is an autoencoder.
        """
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.training_loss(
            outputs,
            features,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(self, features):
        """
        Runs a validation batch where features = targets since this is an autoencoder.
        """
        component_outputs = self.model.encode(features)
        outputs = self.model.decode(component_outputs)

        loss = self.validation_loss(outputs, features)
        self.epoch_validation_loss += loss

    def _run_epoch(self, epoch):
        """
        Since this is an autoencoder, there are no targets and thus the dataloaderss only have features.
        """
        self.training_dataloader.sampler.set_epoch(epoch)

        tic1 = datetime.now()
        self.model.train()
        for features in self.training_dataloader:
            features = features[0].to(self.device, non_blocking=True)
            self._run_training_batch(features)

        tic2 = datetime.now()
        self.model.eval()
        with torch.no_grad():
            for features in self.validation_dataloader:
                features = features[0].to(self.device, non_blocking=True)
                self._run_validation_batch(features)

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}\n")


class EmulatorTrainerSequential(Trainer):
    def __init__(
        self,
        GeneralConfig,
        AEConfig,
        EMConfig,
        loss_functions,
        processing_functions,
        autoencoder: torch.nn.Module,
        emulator: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> None:
        """
        Initializes the EmulatorTrainer, a subclass of Trainer, specialized for train the emulator.
        """
        self.ae = autoencoder
        self.training_loss = loss_functions.training
        self.validation_loss = loss_functions.validation
        self.latent_dim = AEConfig.latent_dim
        self.gradient_clipping = EMConfig.gradient_clipping
        self.inverse_latent_components_scaling = (
            processing_functions.inverse_latent_components_scaling
        )

        super().__init__(
            GeneralConfig,
            model_config=EMConfig,
            model=emulator,
            optimizer=optimizer,
            scheduler=scheduler,
            training_dataloader=training_dataloader,
            validation_dataloader=validation_dataloader,
        )

    def _run_training_batch(self, phys, features, targets):
        """
        Runs a single training batch.
        """
        self.optimizer.zero_grad()

        outputs = self.model(phys, features)

        outputs = outputs.reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
        targets = targets.reshape(-1, 333)

        loss = self.training_loss(outputs, targets, emulator=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.optimizer.step()

    def _run_validation_batch(self, phys, features, targets):
        """
        Runs a single validation batch.
        """

        outputs = self.model(phys, features)

        outputs = outputs.reshape(-1, self.latent_dim)
        outputs = self.inverse_latent_components_scaling(outputs)
        outputs = self.ae.decode(outputs)
        outputs = outputs.reshape(targets.size(0), targets.size(1), -1)

        loss = self.validation_loss(outputs, targets, emulator=True).mean(dim=0)

        self.epoch_validation_loss += loss.detach()

    def _run_epoch(self, epoch):
        """
        Runs a single epoch of training and validation, profiling the first 10 training batches.
        """
        self.training_dataloader.sampler.set_epoch(epoch)
        tic1 = datetime.now()

        self.model.train()

        for phys, features, targets in self.training_dataloader:
            phys = phys.to(self.device, non_blocking=True)
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self._run_training_batch(phys, features, targets)

        tic2 = datetime.now()

        self.model.eval()
        with torch.no_grad():
            for phys, features, targets in self.validation_dataloader:
                phys = phys.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                self._run_validation_batch(phys, features, targets)

        toc = datetime.now()
        print(f"Training Time: {tic2 - tic1} | Validation Time: {toc - tic2}")


def load_objects(model, config):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
        fused=True,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_decay,
        patience=config.lr_decay_patience,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    return optimizer, scheduler
