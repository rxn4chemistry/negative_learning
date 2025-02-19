import logging
import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# PyTorch models inherit from torch.nn.Module
class Baseline(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.0):
        super(Baseline, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.linear(x)


class BaselineSigmoid(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.0):
        super(BaselineSigmoid, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.linear(x)


class BaselineTrainer:
    def __init__(
        self,
        model: Union[Baseline, BaselineSigmoid],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        oversampling_threshold: float = 0.1,
        device=None,
    ):
        self.device = device
        self.oversampling_threshold = oversampling_threshold
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss = nn.MSELoss(reduction="mean")
        self.loss_history = []  # To store the loss value after each update
        self.validation_loss_history = []
        self.validation_samples = None

    def train_step(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        mask: Optional[torch.Tensor] = None,
        batch_size: int = 10000,
    ):
        self.model.train()

        # Removing masked input
        if mask is not None:
            non_zero_mask = mask != 0
            non_zero_indices = non_zero_mask.nonzero()
            x = x[non_zero_indices]
            y = y[non_zero_indices]

        # Checks on the batch composition
        positive_mask = (y > 0.5) if isinstance(self.model, BaselineSigmoid) else y > 0.5
        positive_indices = positive_mask.nonzero()
        n_samples = len(y)
        n_positives = len(y[positive_indices])
        n_negatives = n_samples - n_positives
        logger.info(f"Number of positives in baseline train epoch: {n_positives}")
        logger.info(f"Number of negatives in baseline train epoch: {n_negatives}")

        if (
            n_positives / n_samples < self.oversampling_threshold
            or n_negatives / n_samples < self.oversampling_threshold
        ):
            logger.info(
                f"Aborting baseline training because one class is less than {self.oversampling_threshold * 100}% present ..."
            )
            return

        # If I have enough for each class (at least 1%) -> Oversample
        logger.info("Oversampling minority dataset in baseline...")
        x, y = self.oversample_minority_dataset(x.squeeze().cpu(), y.squeeze().cpu())

        # Shuffling
        rand_indices = torch.randperm(x.size()[0])
        x = x[rand_indices].float().to(self.device)
        y = y[rand_indices].float().view(1, -1).to(self.device)

        tot_steps = math.ceil(x.size()[0] / batch_size)
        logger.info(f"Baseline model trained for: {tot_steps}")
        self.model.train()
        epoch_loss = []
        for step in range(tot_steps):
            start = step * batch_size
            x_batch = x[start : start + batch_size]
            y_batch = y[:, start : start + batch_size]
            y_pred = self.model(x_batch).view(1, -1)
            loss = self.loss(y_pred, y_batch)

            epoch_loss.append(float(loss))
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # update weights
            self.optimizer.step()
        self.loss_history.append(np.mean(epoch_loss))

        if self.validation_samples is not None:
            logger.info("Validating baseline model...")
            self.model.eval()
            x_val, y_val = self.validation_samples
            y_pred = self.model(x_val).view(1, -1)
            loss_valid = self.loss(y_pred, y_val)
            self.validation_loss_history.append(float(loss_valid))

        logger.info("Updating validation samples ...")
        self.validation_samples = (x.detach(), y.detach())

    def preprocess_step(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Issue: if the inputs change continuously I normalize every batch differently
        return f.normalize(x).float()

    def full_step(
        self,
        x: torch.FloatTensor,
        y: torch.FloatTensor,
        mask: Optional[torch.Tensor] = None,
    ):
        x_norm = self.preprocess_step(x)
        self.train_step(x_norm, y, mask)

    def predict(self, x: torch.FloatTensor, randomic: bool = False) -> torch.FloatTensor:
        self.model.eval()
        x = x.float().to(self.device)
        if randomic:
            return torch.rand(x.size()[0:-1]).to(self.device).detach()
        return self.model(x).detach()

    def oversample_minority_dataset(self, x: torch.FloatTensor, y: torch.FloatTensor):
        """
        Function to create a dataset where the minority class is oversampled.
        """
        positive_mask = (y > 0.5) if isinstance(self.model, BaselineSigmoid) else y > 0.5
        negative_mask = torch.logical_not(positive_mask)
        positive_indices = positive_mask.nonzero()
        negative_indices = negative_mask.nonzero()
        n_samples = len(y)
        n_positives = len(y[positive_indices])
        n_negatives = n_samples - n_positives

        y_to_oversample = y[positive_indices] if n_positives < n_negatives else y[negative_indices]
        x_to_oversample = x[positive_indices] if n_positives < n_negatives else x[negative_indices]
        y_majority = y[positive_indices] if n_positives > n_negatives else y[negative_indices]
        x_majority = x[positive_indices] if n_positives > n_negatives else x[negative_indices]
        new_y = y_to_oversample.clone()
        new_x = x_to_oversample.clone()
        minority_dataset_length = len(new_y)
        majority_dataset_length = n_samples - minority_dataset_length
        new_length = minority_dataset_length
        while new_length < majority_dataset_length:
            if len(new_y) + minority_dataset_length <= majority_dataset_length:
                # just add the whole minority dataset
                new_y = torch.cat((new_y, y_to_oversample), dim=0)
                new_x = torch.cat((new_x, x_to_oversample), dim=0)
            else:
                n_samples = majority_dataset_length - len(new_y)
                indices = torch.randint(
                    0, minority_dataset_length, (n_samples,)
                )  # selects n_samples indices at random
                new_y = torch.cat(
                    (new_y, torch.index_select(y_to_oversample, 0, indices)), dim=0
                )  # or dim=0?
                new_x = torch.cat((new_x, torch.index_select(x_to_oversample, 0, indices)), dim=0)

            new_length = len(new_y)
        return torch.cat((new_x, x_majority), dim=0).squeeze(), torch.cat(
            (new_y, y_majority), dim=0
        ).squeeze()


BASELINE_MODELS = {"basic": Baseline, "sigmoid": BaselineSigmoid}
