import itertools
import json
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path

import click
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.models.tokenization import BasicSmilesTokenizer
from rxn_negative_learning.utils.smiles_utils import (
    partial_smiles_sequences,
    randomize_multiple_smiles_rotated,
)

logger = logging.getLogger(__name__)


# files = [
#     "/Users/ato/Desktop/Git/rxn_negative_learning/data/regiosqmv4/decreasingpos/sratio0.3_random_seed42_k0/all/data-train-with-valid-all.jsonl",
#     "/Users/ato/Desktop/Git/rxn_negative_learning/data/regiosqmv4/decreasingpos/sratio0.3_random_seed42_k0/all/data-test-all.jsonl"
# ]


class LookupMode(Enum):
    statistical = "statistical"
    optimistic = "optimistic"


@click.command()
@click.option("-f", "--files_list_jsonl", type=str, required=True, multiple=True)
@click.option("--output_file", type=str, required=True)
@click.option("--augment", type=bool, is_flag=True, default=False)
@click.option("--augmentation_number", type=int, default=3)
@click.option(
    "--lookup_mode", type=click.Choice(["statistical", "optimistic"]), default="statistical"
)
def main(
    files_list_jsonl: list[str],
    output_file: str,
    augment: bool,
    augmentation_number: int,
    lookup_mode: str,
):
    setup_console_logger()

    output_path = Path(output_file)

    logger.info(f"Output path for the dataset conversion: {str(output_path)}")

    tokenizer = BasicSmilesTokenizer()
    lookup_mode = LookupMode(lookup_mode)

    BASELINE_TARGETS_COUNT = defaultdict(int)
    BASELINE_TARGETS_SCORE = defaultdict(int)

    for ff in files_list_jsonl:
        with open(ff, "r") as f:
            data = [json.loads(line.strip()) for line in f]
            for elem in data:
                elems_list = [elem["target"]]
                if augment:
                    for i in range(augmentation_number):
                        new_smi = randomize_multiple_smiles_rotated(elem["target"])
                        elems_list.append(new_smi)
                for smi in elems_list:
                    for p in partial_smiles_sequences(" ".join(tokenizer.tokenize(smi))):
                        conditioned_p = f"{elem['source']}>>{p}"
                        if lookup_mode == lookup_mode.statistical:
                            BASELINE_TARGETS_COUNT[conditioned_p] += 1
                            BASELINE_TARGETS_SCORE[conditioned_p] += elem["score"]
                        elif lookup_mode == lookup_mode.optimistic:
                            BASELINE_TARGETS_COUNT[conditioned_p] = (
                                1  # set to 1 to be compatible with the rest of the code
                            )
                            if elem["score"] == 1:
                                BASELINE_TARGETS_SCORE[conditioned_p] = elem["score"]

    BASELINE_TARGETS_DICT = {}
    for elem in BASELINE_TARGETS_COUNT.keys():
        BASELINE_TARGETS_DICT[elem] = {
            "score": BASELINE_TARGETS_SCORE[elem],
            "count": BASELINE_TARGETS_COUNT[elem],
        }

    print(f"BASELINE_LOOKUP TABLE DIM: {len(BASELINE_TARGETS_DICT.keys())}")
    print(
        "BASELINE LOOKUP TABLE SAMPLE: ", dict(itertools.islice(BASELINE_TARGETS_DICT.items(), 30))
    )

    with open(output_path, "w") as f:
        json.dump(BASELINE_TARGETS_DICT, f, indent=4)


if __name__ == "__main__":
    main()
