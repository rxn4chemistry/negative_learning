import json

import pytest
import pytorch_lightning as pl
import torch
from rxn_negative_learning.models.scorers.scorer_base import RXNNegScorerBase
from rxn_negative_learning.utils.repo_utils import reset_random_seed


def set_randomness():
    pl.seed_everything(42)
    reset_random_seed()
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def files(tmp_path_factory):
    vocabulary = [
        "[PAD]",
        "[unused1]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "c",
        "C",
        "N",
        "-",
        "(",
        ")",
        "O",
        "Na",
        "Cl",
        "Mg",
        "Br",
        ".",
        "1",
        "=",
        "n",
        "H",
        "[",
        "]",
    ]

    data = [
        {
            "source": "O=C1CCC(=O)N1Br.c1cn[nH]c1",
            "target": "Brc1cn[nH]c1",
            "score": 1,
            "idx": 0,
            "opposite_targets": ["Brc1cc[nH]n1", "Brc1ccn[nH]1"],
        },
        {
            "source": "O=C1CCC(=O)N1Br.c1cn[nH]c1",
            "target": "Brc1cc[nH]n1",
            "score": 0,
            "idx": 0,
            "opposite_targets": "Brc1cn[nH]c1",
        },
        {
            "source": "BrBr.c1cnsc1",
            "target": "Brc1cnsc1",
            "score": 1,
            "idx": 9,
            "opposite_targets": ["Brc1ccns1", "Brc1ccsn1"],
        },
        {
            "source": "Cn1cncc1-c1ccccc1.O=C1CCC(=O)N1Br",
            "target": "Cn1cnc(Br)c1-c1ccccc1",
            "score": 1,
            "idx": 14,
            "opposite_targets": [
                "Cn1c(-c2ccccc2)cnc1Br",
                "Cn1cncc1-c1ccc(Br)cc1",
                "Cn1cncc1-c1cccc(Br)c1",
                "Cn1cncc1-c1ccccc1Br",
            ],
        },
        {
            "source": "Cn1cncc1-c1ccccc1.O=C1CCC(=O)N1Br",
            "target": "Cn1c(-c2ccccc2)cnc1Br",
            "score": 0,
            "idx": 14,
            "opposite_targets": "Cn1cnc(Br)c1-c1ccccc1",
        },
    ]
    data = [json.dumps(elem) for elem in data]

    tmp_vocab_file = tmp_path_factory.mktemp("data") / "vocab.txt"
    with open(tmp_vocab_file, "w") as f:
        f.write("\n".join(vocabulary))

    tmp_data_file = tmp_path_factory.mktemp("data") / "data.jsonl"
    with open(tmp_data_file, "w") as f:
        f.write("\n".join(data))

    return {
        "vocab": tmp_vocab_file,
        "train": tmp_data_file,
        "test": tmp_data_file,
        "validation": tmp_data_file,
    }


@pytest.fixture(scope="session")
def parameters(files):
    model_args = {
        "model_name_or_path": None,
        "model_config_name": None,
        "vocabulary": files["vocab"],
        "train_file": files["train"],
        "test_file": files["test"],
        "validation_file": files["validation"],
        "num_dataloader_workers": 1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.998,
        "adam_epsilon": 1.0e-09,
        "adam_weight_decay": 0.01,
        "embedding_dim": 8,
        "ffnn_hidden_dim": 3,
        "dropout": 0.0,
        "activation": "relu",
        "num_attention_heads": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "init_std": 0.02,
        "max_position_embeddings": 5000,
        "num_beams": 1,
        "learning_rate": 1e-4,
        "no_teacher_forcing": True,
        "with_baseline": False,
        "baseline_targets_file": None,
        "regularization_beta": 1,
        "regularization": None,
        "max_steps": 2,
        "augment": False,
        "batch_size": 3,
        "lr_scheduler": "constant",
        "save_validation_predictions": False,
        "save_train_predictions": False,
        "update_regularization_model_every_step": None,
        "regularization_beta_clipping": None,
        "baseline_model_dropout": 0.01,
        "baseline_model_lr": 1e-5,
        "baseline_model_weight_decay": 0.01,
        "baseline_model_oversampling_threshold": 0.01,
        "randomic": False,
        "baseline_batch_size": 100,
        "update_lookup_table_every_n_epochs": 1,
    }
    return model_args


@pytest.fixture(scope="session")
def dummy_scorer():
    pos_samples = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1", "BrBr.c1cnsc1>>Brc1cnsc1"]
    neg_samples = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cc[nH]n1"]

    class DummyScorer(RXNNegScorerBase):
        # Needed otherwise the baseline training is skipped
        def __init__(self, positive_reactions=None, negative_reactions=None, shift: float = 0.0):
            super().__init__(positive_reactions, negative_reactions, shift)

        def score(self, reactions):
            rewards = [0.0 for _ in range(len(reactions))]
            rewards[0] = 1.0
            return rewards

    return DummyScorer(pos_samples, neg_samples)
