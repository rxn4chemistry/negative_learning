from argparse import Namespace

import pytest
import pytorch_lightning as pl
import torch
from rxn_negative_learning.models.base_transformer.pytorch_lightning.dataset import LitSmilesDataset
from rxn_negative_learning.models.rl_transformer.reinforce_lightning import (
    ReinforceLitVanillaTransformer,
)
from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.utils.repo_utils import tests_directory
from torch.nn.functional import log_softmax

from tests.conftest import set_randomness


def test_regularization(parameters, dummy_scorer):
    """Tests the ability to make training step"""
    set_randomness()

    tokenizer = SmilesTokenizer(parameters["vocabulary"])
    parameters["regularization"] = "kl-reverse"
    rl_model = ReinforceLitVanillaTransformer(
        model_args=parameters,
        tokenizer=tokenizer,
        scorer=dummy_scorer,
    )

    BATCH_SIZE = 3
    MAX_LENGTH = 6
    VOCAB_SIZE = 4  # tokenizer.vocab_size

    # (target - input)
    base_log_probs = log_softmax(torch.rand((BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE)), dim=-1)
    logits = torch.rand((BATCH_SIZE, MAX_LENGTH, VOCAB_SIZE))
    log_probs = log_softmax(logits, dim=-1)
    pred_ids = torch.argmax(logits, dim=-1).detach()
    end_of_sequence_mask = torch.zeros((BATCH_SIZE, MAX_LENGTH))
    end_of_sequence_mask[:, 0:-2] = 1
    end_of_sequence_mask[2, MAX_LENGTH - 4 :] = 0
    end_of_sequence_mask = torch.reshape(
        end_of_sequence_mask, (end_of_sequence_mask.shape[-1] * end_of_sequence_mask.shape[0],)
    )

    # KL
    assert rl_model.compute_regularization_loss(
        base_log_probs, log_probs, pred_ids, end_of_sequence_mask
    ).item() == pytest.approx(0.051488328725099564)

    # KL abs
    parameters["regularization"] = "kl-abs"
    rl_model = ReinforceLitVanillaTransformer(
        model_args=parameters,
        tokenizer=tokenizer,
        scorer=dummy_scorer,
    )
    assert rl_model.compute_regularization_loss(
        base_log_probs, log_probs, pred_ids, end_of_sequence_mask
    ).item() == pytest.approx(0.26473158597946167)

    # JSD
    parameters["regularization"] = "jsd"
    rl_model = ReinforceLitVanillaTransformer(
        model_args=parameters,
        tokenizer=tokenizer,
        scorer=dummy_scorer,
    )
    assert rl_model.compute_regularization_loss(
        base_log_probs, log_probs, pred_ids, end_of_sequence_mask
    ).item() == pytest.approx(0.05154521018266678)


def test_model_with_regularization(parameters, dummy_scorer):
    """Tests the ability to make training step"""
    set_randomness()

    tokenizer = SmilesTokenizer(parameters["vocabulary"])
    parameters["regularization"] = "kl-reverse"
    rlmodel = ReinforceLitVanillaTransformer(
        model_args=parameters,
        tokenizer=tokenizer,
        scorer=dummy_scorer,
    )

    # creating dummy input and output of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length + 3)).long() * 7
    output_ids = torch.ones((batch_size, sequence_length)).long()

    output = rlmodel.training_step(
        {
            "encoder_input_ids": input_ids,
            "decoder_input_ids": output_ids,
            "score": torch.ones((batch_size,)).long(),
            "idx": torch.ones((batch_size,)).long(),
        },
        batch_idx=0,
    )

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

    assert output.detach() == pytest.approx(2.384185791015625e-07)


def test_model_with_baseline_targets_file(parameters, dummy_scorer):
    """Tests the ability to make training step"""
    set_randomness()

    tokenizer = SmilesTokenizer(parameters["vocabulary"])
    parameters["baseline_targets_file"] = str(tests_directory() / "baseline_targets_test.json")
    parameters["with_baseline"] = "sigmoid"
    rlmodel = ReinforceLitVanillaTransformer(
        model_args=parameters,
        tokenizer=tokenizer,
        scorer=dummy_scorer,
    )

    # creating dummy input and output of size (batch_size, sequence_length)
    sequence_length = 10
    batch_size = 2
    input_ids = torch.ones((batch_size, sequence_length + 3)).long() * 7
    output_ids = torch.ones((batch_size, sequence_length)).long()

    output = rlmodel.training_step(
        {
            "encoder_input_ids": input_ids,
            "decoder_input_ids": output_ids,
            "score": torch.ones((batch_size,)).long(),
            "idx": torch.ones((batch_size,)).long(),
        },
        batch_idx=0,
    )

    print(output.detach())
    assert output.detach() == pytest.approx(-0.0066645145416259766)


def test_update_lookup_table(parameters, dummy_scorer):
    set_randomness()
    tokenizer = SmilesTokenizer(parameters["vocabulary"])
    parameters["baseline_targets_file"] = str(tests_directory() / "baseline_targets_test.json")
    parameters["with_baseline"] = "sigmoid"
    parameters["max_steps"] = 4
    parameters["update_lookup_table_every_n_epochs"] = 2

    # Create model
    rlmodel = ReinforceLitVanillaTransformer(
        model_args=parameters, tokenizer=tokenizer, scorer=dummy_scorer
    )
    old_lookup = rlmodel.BASELINE_TARGETS_DICT.copy()

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

    new_lookup = trainer.model.BASELINE_TARGETS_DICT.copy()
    assert old_lookup != new_lookup
