import logging
from pathlib import Path
from typing import Union

import click
import pandas as pd
from rxn.chemutils.reaction_equation import ReactionEquation, rxn_standardization
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.scripts.extract_buitrago_data import OUTPUT_CSV, load_buitrago_data

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

INPUT_CSV = OUTPUT_CSV
OUTPUT_FILE_NAME = "buitrago.csv"


def generate_buitrago_negatives(
    input_csv: Union[str, Path], output_csv: Union[str, Path], threshold: float
):
    """
    Script to separate positives and negatives for the buitragoOLD dataset

    input_csv: input csv file, the first column contains the reaction and the second the yield/amount-of-product information.
    output_csv: where the splits will be stored in two different csv files
    threshold: the cut for the yield/amount-of-product were to discern between positives and negatives
    """
    df = pd.read_csv(input_csv)
    rxns_column = df.columns[0]
    yield_column = df.columns[1]
    logger.info(f"Reactions column: {rxns_column}")
    logger.info(f"Yield column: {yield_column}")

    df["score"] = df[yield_column].apply(lambda x: "1" if (x > threshold) else "0")
    logger.info(f"Number of positives: {len(df.score.loc[df.score == '1'])}")
    logger.info(f"Number of negatives: {len(df.score.loc[df.score == '0'])}")
    logger.info(df.head())

    df[rxns_column] = df[rxns_column].apply(
        lambda x: rxn_standardization(ReactionEquation.from_string(x)).to_string()
    )
    df.to_csv(output_csv, index=False)


@click.command()
@click.argument("output_folder", type=str, required=True)
@click.option("--threshold", "-t", type=float, default=0.0)
def main(output_folder: str, threshold: float):
    setup_console_logger()
    output_dir = Path(output_folder)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if any(Path(output_dir).iterdir()):
            raise FileExistsError("The output directory already exists and is not empty")
    output_csv = output_dir / OUTPUT_FILE_NAME

    logger.info("***DATA EXTRACTION***")
    load_buitrago_data()
    logger.info("***DATA FORMATTING AND NEGATIVES GENERATION***")
    generate_buitrago_negatives(INPUT_CSV, output_csv, threshold)
    logger.info("***DONE!***")


if __name__ == "__main__":
    main()
