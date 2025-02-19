import logging
from pathlib import Path
from typing import Union

import click
import numpy as np
import pandas as pd
from rxn.chemutils.exceptions import InvalidSmiles
from rxn.chemutils.reaction_equation import (
    ReactionEquation,
    remove_precursors_from_products,
    rxn_standardization,
)
from rxn.chemutils.utils import remove_atom_mapping
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.utils.repo_utils import data_directory

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

INPUT_CSV = f"{data_directory() / 'source_data' / 'uspto' / '1976_Sep2016_USPTOgrants_smiles.rsmi'}"
OUTPUT_FILE_NAME = "uspto.csv"


def process_uspto_dataset(input_csv: Union[str, Path], output_csv: Union[str, Path]):
    """
    Script to process the USPTO dataset

    input_csv: input file with the first column being the reactions
    output_csv: output file where the cleaned reactions will be stored
    """
    df = pd.read_csv(input_csv, sep="\t")
    rxns_column = df.columns[0]
    logger.info(df.head())
    logger.info(f"Number of reactions: {len(df)}")
    logger.info("Removing duplicate reactions ...")
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Number of reactions: {len(df)}")

    def standardize(x: str) -> str:
        x = remove_atom_mapping(x)
        reaction_equation = ReactionEquation.from_string(x)
        if len(reaction_equation.products) != 1:  # remove single product reactions
            return ""
        try:
            reaction_equation = remove_precursors_from_products(
                rxn_standardization(ReactionEquation.from_string(x))
            )
            reaction_equation_one_product = (
                "" if len(reaction_equation.products) != 1 else reaction_equation.to_string()
            )
            return reaction_equation_one_product
        except InvalidSmiles:
            return ""

    df["rxn"] = df[rxns_column].apply(lambda x: standardize(x))
    logger.info("Removing rxns failing standardization ...")
    df = df[df["rxn"] != ""]
    df = df.drop_duplicates(subset=["rxn"]).reset_index(drop=True)
    df = df.replace(np.nan, "", regex=True)
    logger.info(f"Number of reactions: {len(df)}")

    df.to_csv(output_csv, index=False)


@click.command()
@click.argument("output_folder", type=str, required=True)
def main(output_folder: str):
    setup_console_logger()
    output_dir = Path(output_folder)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if any(Path(output_dir).iterdir()):
            raise FileExistsError("The output directory already exists and is not empty")
    output_csv = output_dir / OUTPUT_FILE_NAME

    logger.info("***DATA PREPARATION***")
    process_uspto_dataset(INPUT_CSV, output_csv)
    logger.info("***DONE!***")


if __name__ == "__main__":
    main()
