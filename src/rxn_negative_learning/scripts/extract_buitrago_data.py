import logging
import re

import pandas as pd
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.data_generation.help_dicts_buitrago import PRODUCTS, SOLVENT, STRUCTURES
from rxn_negative_learning.utils.repo_utils import data_directory

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

INPUT_FILE = f"{data_directory() / 'source_data' / 'buitrago' / '1259203_datafiles.xlsx'}"
OUTPUT_CSV = f"{data_directory() / 'source_data' / 'buitrago' / 'extracted_buitrago_data.csv'}"

# Name to give to the new columns containing only the structure label
BASE_COLUMN = "base_label"
CATALYST_COLUMN = "catalyst_label"
ELECTROPHILE_COLUMN = "electrophile_label"
NUCLEOPHILE_COLUMN = "nucleophile_label"


def create_rxn_smiles(x: pd.Series) -> pd.Series:
    """
    Function to create rxn smiles from the columns of a Pandas DataFrame.

    To be used with DataFrame.apply().
    """

    base_smiles = STRUCTURES[x[BASE_COLUMN]]
    catalyst_smiles = STRUCTURES[x[CATALYST_COLUMN]]
    electrophile_smiles = STRUCTURES[x[ELECTROPHILE_COLUMN]]
    nucleophile_smiles = STRUCTURES[x[NUCLEOPHILE_COLUMN]]
    solvent_smiles = SOLVENT

    # the products above have the same labels as the nucleophiles
    product_smiles = PRODUCTS[x[NUCLEOPHILE_COLUMN]]

    return pd.Series([
        f"{solvent_smiles}.{electrophile_smiles}.{nucleophile_smiles}.{base_smiles}.{catalyst_smiles}>>{product_smiles}"
    ])


def get_label(full_name: str) -> str:
    """
    Get the label of a structure from the string present in the XLSX.

    Example: "XantPhos Pd G2 32" -> "32".
    """
    regex_string = r" (S?\d+)"
    m = re.search(regex_string, full_name)
    if m is None:
        raise RuntimeError("no match")
    return m.group(1)


def load_buitrago_data():
    """Extract the reactions from the Supporting Information of the Buitrago
    Santanilla paper (10.1126/science.1259203).
    """

    # Read from the XLSX file, by specifying the name of the sheet.
    df = pd.read_excel(INPUT_FILE, sheet_name="Data S2- Experiment 2")

    logger.info(f"Loaded XLSX sheet. Number entries: {len(df)}")

    df.dropna(subset=["Pd/IS"], inplace=True)
    logger.info(f"After removal of nan values: {len(df)}")

    df[BASE_COLUMN] = df["Base"].apply(get_label)
    df[CATALYST_COLUMN] = df["Catalyst"].apply(get_label)
    df[ELECTROPHILE_COLUMN] = df["Electrophile"].apply(get_label)
    df[NUCLEOPHILE_COLUMN] = df["Nucleophile"].apply(get_label)

    logger.info(df.head())
    df["rxn"] = df.apply(create_rxn_smiles, axis=1)

    new_df: pd.DataFrame = df[["rxn", "Pd/IS"]].copy()
    new_df.to_csv(OUTPUT_CSV, index=False)


def main():
    setup_console_logger()
    load_buitrago_data()


if __name__ == "__main__":
    main()
