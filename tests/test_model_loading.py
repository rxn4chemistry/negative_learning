from argparse import Namespace

import pytorch_lightning as pl
import torch
from rxn_negative_learning.models.base_transformer.pytorch_lightning.dataset import LitSmilesDataset
from rxn_negative_learning.models.baseline.baseline_model import BaselineTrainer
from rxn_negative_learning.models.rl_transformer.reinforce_lightning import (
    ReinforceLitVanillaTransformer,
)
from rxn_negative_learning.models.scorers.ideal_scorer import IdealScorer
from rxn_negative_learning.models.tokenization import SmilesTokenizer

from tests.conftest import set_randomness


def test_model_init_simple(parameters):
    set_randomness()
    tokenizer = SmilesTokenizer(parameters["vocabulary"])
    pos_samples = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1", "BrBr.c1cnsc1>>Brc1cnsc1"]
    neg_samples = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cc[nH]n1"]

    # Create model
    rlmodel = ReinforceLitVanillaTransformer(
        model_args=parameters, tokenizer=tokenizer, scorer=IdealScorer(pos_samples, neg_samples)
    )
    assert rlmodel.baseline_model is None
    assert rlmodel.BASELINE_TARGETS_DICT is None
    assert isinstance(rlmodel.scorer, IdealScorer)
    assert rlmodel.reference_model is None

    # Create Dataset
    smiles_dataset = LitSmilesDataset(dataset_args=parameters, tokenizer=tokenizer)
    smiles_dataset.load()
    train_dataloader = smiles_dataset.train_dataloader()
    val_dataloader = smiles_dataset.val_dataloader()

    # Create trainer
    n_parameters = Namespace(**parameters)
    trainer = pl.Trainer.from_argparse_args(
        n_parameters,
        deterministic=True,
    )
    trainer.fit(
        rlmodel,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.save_checkpoint("/tmp/checkpoint.ckpt")

    # For finetuning
    parameters["regularization_beta"] = 2
    new_rlmodel = ReinforceLitVanillaTransformer.load_from_checkpoint(
        "/tmp/checkpoint.ckpt", model_args=parameters, strict=False
    )
    assert new_rlmodel.baseline_model is None
    assert new_rlmodel.BASELINE_TARGETS_DICT is None
    assert isinstance(new_rlmodel.scorer, IdealScorer)
    assert new_rlmodel.reference_model is None
    assert new_rlmodel.regularization_beta == 2

    # For resuming
    model = ReinforceLitVanillaTransformer(
        model_args=parameters, tokenizer=tokenizer, scorer=IdealScorer(pos_samples, neg_samples)
    )
    trainer = pl.Trainer.from_argparse_args(
        n_parameters,
        deterministic=True,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path="/tmp/checkpoint.ckpt",
    )


def test_model_init_with_baseline(parameters, dummy_scorer):
    set_randomness()
    tokenizer = SmilesTokenizer(parameters["vocabulary"])
    pos_samples = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1", "BrBr.c1cnsc1>>Brc1cnsc1"]
    neg_samples = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cc[nH]n1"]

    # Adding baseline-related parameters
    parameters["with_baseline"] = "sigmoid"
    parameters["baseline_model_dropout"] = 0.01
    parameters["baseline_model_lr"] = 1e-5
    parameters["baseline_model_oversampling_threshold"] = 0.1
    parameters["baseline_model_weight_decay"] = 1e-3
    parameters["randomic"] = False
    parameters["baseline_batch_size"] = 10000

    # Create model
    rlmodel = ReinforceLitVanillaTransformer(
        model_args=parameters, tokenizer=tokenizer, scorer=dummy_scorer
    )
    assert isinstance(rlmodel.baseline_model, BaselineTrainer)
    assert rlmodel.BASELINE_TARGETS_DICT is None
    assert rlmodel.reference_model is None

    # Create Dataset
    smiles_dataset = LitSmilesDataset(dataset_args=parameters, tokenizer=tokenizer)
    smiles_dataset.load()
    train_dataloader = smiles_dataset.train_dataloader()
    val_dataloader = smiles_dataset.val_dataloader()

    # Create trainer
    n_parameters = Namespace(**parameters)
    trainer = pl.Trainer.from_argparse_args(
        n_parameters,
        deterministic=True,
    )
    trainer.fit(
        rlmodel,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    trainer.save_checkpoint("/tmp/checkpoint.ckpt")

    # For finetuning
    new_rlmodel = ReinforceLitVanillaTransformer.load_from_checkpoint(
        "/tmp/checkpoint.ckpt", scorer=dummy_scorer, strict=False
    )
    isinstance(rlmodel.baseline_model, BaselineTrainer)
    assert new_rlmodel.BASELINE_TARGETS_DICT is None
    assert new_rlmodel.reference_model is None

    # For resuming
    parameters["max_steps"] = 4
    n_parameters = Namespace(**parameters)
    new_new_rlmodel = ReinforceLitVanillaTransformer(
        model_args=parameters, tokenizer=tokenizer, scorer=IdealScorer(pos_samples, neg_samples)
    )
    trainer = pl.Trainer.from_argparse_args(
        n_parameters,
        deterministic=True,
    )

    m = torch.load("/tmp/checkpoint.ckpt")
    print(f"{m.keys()}")
    print(f"Epoch: {m['epoch']}")
    print(f"Global step: {m['global_step']}")
    print(m["reference_model"])
    print(str(m["baseline_model"]))

    trainer.fit(
        new_new_rlmodel,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path="/tmp/checkpoint.ckpt",
    )
    # This works because in the new_new_rl_model the baseline is not training due to only negative samples
    assert all([
        torch.eq(i, j).all()
        for i, j in zip(
            rlmodel.baseline_model.model.state_dict().values(),
            new_new_rlmodel.baseline_model.model.state_dict().values(),
        )
    ])
