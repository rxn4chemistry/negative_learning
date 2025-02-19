import random
from pathlib import Path

import numpy
import pytorch_lightning as pl
import torch


def root_directory() -> Path:
    """
    Returns the path to the root directory of the repository
    """
    return Path(__file__).parent.parent.parent.parent.resolve()


def data_directory() -> Path:
    """
    Returns the path to the data directory at the root of the repository
    """
    return root_directory() / "data"


def models_directory() -> Path:
    """
    Returns the path to the data directory at the root of the repository
    """
    return root_directory() / "models"


def tests_directory() -> Path:
    """
    Returns the path to the tests directory at the root of the repository
    """
    return root_directory() / "tests"


def notebooks_directory() -> Path:
    """
    Returns the path to the notebooks directory at the root of the repository
    """
    return root_directory() / "notebooks"


def reset_random_seed() -> None:
    random.seed(42)
    numpy.random.seed(42)
    pl.seed_everything(42)
    torch.manual_seed(42)
