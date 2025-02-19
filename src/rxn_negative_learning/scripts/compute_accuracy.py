import json
import logging

import click
from rxn.chemutils.tokenization import detokenize_smiles
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.metrics import top_n_accuracy, top_n_accuracy_neg, top_n_invalids
from rxn_negative_learning.utils.smiles_utils import lazy_canonicalize_smiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.argument("predictions_file", type=str, required=True)
@click.option("--target_file_txt", type=str, default=None)
@click.option("--target_file_jsonl", type=str, default=None)
def main(predictions_file: str, target_file_txt: str, target_file_jsonl: str):
    setup_console_logger()

    if target_file_txt is None and target_file_jsonl is None:
        raise ValueError("At least one target file as `txt` or `jsonl` should be provided!")

    with open(predictions_file, "r") as f:
        predictions = [lazy_canonicalize_smiles(detokenize_smiles(line.strip())) for line in f]
        logger.info(f"Number of predictions: {len(predictions)}")
    if target_file_txt is not None:
        with open(target_file_txt, "r") as f:
            targets = [lazy_canonicalize_smiles(detokenize_smiles(line.strip())) for line in f]
    else:
        with open(target_file_jsonl, "r") as f:
            data = [json.loads(line.strip()) for line in f]
            logger.info(f"Number of targets: {len(data)}")
            targets = [
                lazy_canonicalize_smiles(detokenize_smiles(data[i]["target"]))
                for i in range(len(data))
            ]

            logger.info("***TOP-N POSITIVE ACCURACY***")
            print(top_n_accuracy(targets, predictions))

            if "opposite_targets" in data[0].keys():
                targets_neg = [
                    [
                        lazy_canonicalize_smiles(detokenize_smiles(datum))
                        for datum in data[i]["opposite_targets"]
                    ]
                    for i in range(len(data))
                ]
                logger.info("***TOP-N NEGATIVE ACCURACY***")
                print(top_n_accuracy_neg(targets_neg, predictions))

    logger.info("***TOP-N INVALIDS RATIO***")
    print(top_n_invalids(targets, predictions))


if __name__ == "__main__":
    main()
