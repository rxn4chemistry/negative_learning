import json
import logging
from collections import Counter
from pathlib import Path
from typing import List

import click
from rxn.chemutils.tokenization import tokenize_smiles
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.utils.smiles_utils import flatten

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

vocab = [
    "[PAD]",
    "[unused1]",
    "[unused2]",
    "[unused3]",
    "[unused4]",
    "[unused5]",
    "[unused6]",
    "[unused7]",
    "[unused8]",
    "[unused9]",
    "[unused10]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
]


@click.command()
@click.argument("output_folder", type=str, required=True)
@click.option("--file_jsonl", multiple=True, type=str)
def main(file_jsonl: List[str], output_folder: str):
    setup_console_logger()
    chars = []
    for f in file_jsonl:
        logger.info(f"Processing file '{f}' ...")
        with open(f, "r") as g:
            data = [json.loads(line) for line in g]
        sources = flatten([tokenize_smiles(data[i]["source"]).split(" ") for i in range(len(data))])
        targets = flatten([tokenize_smiles(data[i]["target"]).split(" ") for i in range(len(data))])
        chars.extend(sources)
        chars.extend(targets)
    for k, _ in sorted(Counter(chars).items(), key=lambda item: item[1], reverse=True):
        vocab.append(k)

    output_file = Path(output_folder)
    output_file.mkdir(exist_ok=True, parents=True)
    output_file = output_file / "vocab.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(vocab))
    logger.info(f"Wrote vocab to: {output_file}")
    return


if __name__ == "__main__":
    main()
