import json

import pytest
import pytorch_lightning as pl
import torch
from rxn_negative_learning.models.base_transformer.pytorch_lightning.dataset import LitSmilesDataset
from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.utils.repo_utils import reset_random_seed


@pytest.fixture(scope="session")
def set_randomness():
    pl.seed_everything(42)
    reset_random_seed()
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def vocab_file(tmp_path_factory):
    vocabulary = [
        "[PAD]",
        "[unused1]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "c",
        "C",
        "(",
        ")",
        "O",
        "N",
        "Br",
        "1",
        "=",
        "n",
        "o",
    ]
    tmp_vocab_file = tmp_path_factory.mktemp("data") / "vocab.txt"
    with open(tmp_vocab_file, "w") as f:
        f.write("\n".join(vocabulary))
    return tmp_vocab_file


@pytest.fixture(scope="session")
def smiles_file(tmp_path_factory):
    smiles_json_lines = [
        {
            "source": "Cc1ccno1.O=C1CCC(=O)N1Br",
            "target": "Cc1oncc1Br",
            "score": 1,
            "idx": 5,
            "opposite_targets": ["Cc1cc(Br)no1"],
        },
        {
            "source": "CC(=O)O.N.Nc1ncccn1.O=C1CCC(=O)N1Br",
            "target": "Nc1ncc(Br)cn1",
            "score": 1,
            "idx": 23,
            "opposite_targets": ["Nc1nccc(Br)n1"],
        },
    ]

    dumped = [json.dumps(line) for line in smiles_json_lines]

    tmp_data_file = tmp_path_factory.mktemp("data") / "smiles.jsonl"
    with open(tmp_data_file, "w") as f:
        f.write("\n".join(dumped))
    return tmp_data_file


def test_dataset_get_item(vocab_file, smiles_file, set_randomness):
    tokenizer = SmilesTokenizer(vocab_file)
    smiles_dataset = LitSmilesDataset(
        dataset_args={
            "num_dataloader_workers": 1,
            "train_file": smiles_file,
            "validation_file": smiles_file,
            "test_file": smiles_file,
            "batch_size": 2,
            "augment": True,
        },
        tokenizer=tokenizer,
    )
    # smiles_dataset = SmilesDataset(filepath=str(smiles_file), tokenizer=tokenizer)
    smiles_dataset.load()
    train_dataloader = smiles_dataset.train_dataloader()
    test_dataloader = smiles_dataset.test_dataloader()
    valid_dataloader = smiles_dataset.val_dataloader()

    # First call: ex. first epoch
    assert train_dataloader.dataset[0]["source_augmented"] == "Cc1ccno1.C1C(=O)N(Br)C(=O)C1"
    assert (
        train_dataloader.dataset[1]["source_augmented"] == "OC(=O)C.N.n1cccnc1N.C1C(=O)N(Br)C(=O)C1"
    )

    # Second call: ex. second epoch
    assert train_dataloader.dataset[0]["source_augmented"] == "c1cc(C)on1.O=C1CCC(=O)N1Br"
    tmp = train_dataloader.dataset[1]
    assert "source_augmented" not in tmp.keys()

    # # Third call would be:
    # assert train_dataloader.dataset[0]["source_augmented"] == "BrN1C(=O)CCC1=O.o1nccc1C"
    # assert train_dataloader.dataset[1]["source_augmented"] == "c1(N)ncccn1.N.OC(=O)C.BrN1C(=O)CCC1=O"
    #
    # But if I start from epoch 0 I get back the initial results
    pl.seed_everything(42)
    reset_random_seed()
    torch.use_deterministic_algorithms(True)

    assert train_dataloader.dataset[0]["source_augmented"] == "Cc1ccno1.C1C(=O)N(Br)C(=O)C1"
    assert (
        train_dataloader.dataset[1]["source_augmented"] == "OC(=O)C.N.n1cccnc1N.C1C(=O)N(Br)C(=O)C1"
    )

    # check that for validation and test no augmentation is performed
    tmp = test_dataloader.dataset[0]
    assert "source_augmented" not in tmp.keys()
    tmp = test_dataloader.dataset[1]
    assert "source_augmented" not in tmp.keys()

    tmp = valid_dataloader.dataset[0]
    assert "source_augmented" not in tmp.keys()
    tmp = valid_dataloader.dataset[1]
    assert "source_augmented" not in tmp.keys()


def test_dataset_get_item_no_augmentation(vocab_file, smiles_file, set_randomness):
    tokenizer = SmilesTokenizer(vocab_file)
    smiles_dataset = LitSmilesDataset(
        dataset_args={
            "num_dataloader_workers": 1,
            "train_file": smiles_file,
            "validation_file": smiles_file,
            "test_file": smiles_file,
            "batch_size": 2,
            "augment": False,
        },
        tokenizer=tokenizer,
    )
    # smiles_dataset = SmilesDataset(filepath=str(smiles_file), tokenizer=tokenizer)
    smiles_dataset.load()
    train_dataloader = smiles_dataset.train_dataloader()

    tmp = train_dataloader.dataset[0]
    assert "source_augmented" not in tmp.keys()
    tmp = train_dataloader.dataset[1]
    assert "source_augmented" not in tmp.keys()
