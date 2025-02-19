import json

from rxn_negative_learning.models.base_transformer.pytorch_lightning.dataset import LitSmilesDataset
from rxn_negative_learning.models.tokenization import SmilesTokenizer


def test_dataloader_with_score(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    VOCAB = [
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
        "Na",
        "Cl",
        "Mg",
    ]
    DATA = [
        {"source": "C.O.CCC(O)", "target": "CCC", "score": 1, "idx": 1},
        {"source": "C.O.CCC(O)", "target": "CCCO.Mg", "score": 0, "idx": 2},
        {"source": "C.O.CCC(O)", "target": "CCCO", "score": 0, "idx": 3},
    ]

    vocab_path = tmp_path / "dummy-vocab.txt"
    vocab_path.write_text("\n".join(VOCAB))
    assert vocab_path.read_text() == "\n".join(VOCAB)

    data_path = tmp_path / "dummy-data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for item in DATA:
            json_record = json.dumps(item, ensure_ascii=False)
            f.write(json_record + "\n")
    out = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line.rstrip("\n|\r")))
    assert out == DATA

    data_args = {
        "train_file": data_path,
        "validation_file": data_path,
        "test_file": data_path,
        "num_dataloader_workers": 1,
        "batch_size": 2,
        "augment": False,
    }

    tokenizer = SmilesTokenizer(vocab_path)

    smiles_dataset = LitSmilesDataset(dataset_args=data_args, tokenizer=tokenizer)
    smiles_dataset.load()
    # print(f"Length of dataloader: {len(smiles_dataset.train_dataloader())}")
    # for batch_idx, sample in enumerate(smiles_dataset.datasets['train']):
    #     print(f"Batch idx: {batch_idx},  Score: {sample['score']}, Target: {sample['target']}")
    assert smiles_dataset.datasets["train"][0]["score"] == 1
    assert len(smiles_dataset.train_dataloader()) == 2
