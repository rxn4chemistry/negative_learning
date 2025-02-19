import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.utils.smiles_utils import (
    get_precursors_from_smiles,
    get_product_from_smiles,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def split_simple(
    input_file_csv: str,
    output_folder: str,
    split_ratio: float,
    reaction_column_name: str,
):
    df = pd.read_csv(input_file_csv)
    df["precursors"] = df[reaction_column_name].apply(lambda x: get_precursors_from_smiles(x))
    df["product"] = df[reaction_column_name].apply(lambda x: get_product_from_smiles(x))
    df["idx"] = [i for i in range(len(df))]  # to perform a random splitting
    logger.info("Splitting dataset ...")

    train_size = 1 - 2 * split_ratio
    validate_size = split_ratio
    train, valid, test = np.split(
        df.sample(frac=1, random_state=42),
        [int(train_size * len(df)), int((validate_size + train_size) * len(df))],
    )
    train_with_valid = pd.concat([train, valid])

    logger.info("Saving data splits ...")
    for label, split_df in zip(
        ["train", "valid", "train-with-valid", "test"],
        [train, valid, train_with_valid, test],
    ):
        converted_lines = []
        for index, row in split_df.iterrows():
            d = {"source": row["precursors"], "target": row["product"]}
            converted_lines.append(d)

        dumped_df = [json.dumps(line) for line in converted_lines]
        output_jsonl = Path(output_folder) / f"data-{label}.jsonl"
        with open(output_jsonl, "w") as f:
            f.write("\n".join(dumped_df))


@click.command()
@click.argument("input_file_csv", type=str, required=True)
@click.argument("output_folder", type=str, required=True)
@click.option("--split_ratio", type=float, default=0.1)
@click.option("--reaction_column_name", type=str, default="rxn")
def main(
    input_file_csv: str,
    output_folder: str,
    split_ratio: float,
    reaction_column_name: str,
):
    setup_console_logger()
    output_dir = Path(output_folder)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if any(Path(output_dir).iterdir()):
            raise FileExistsError("The output directory already exists and is not empty")

    logger.info("***DATA PREPARATION***")
    split_simple(input_file_csv, output_folder, split_ratio, reaction_column_name)
    logger.info("***DONE!***")


if __name__ == "__main__":
    main()
