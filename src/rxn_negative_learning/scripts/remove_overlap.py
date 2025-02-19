import logging
from pathlib import Path
from typing import List, Union

import click
import pandas as pd
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def remove_overlap(
    reference_dataset_csv: Union[str, Path],
    input_files_csv: List[Union[str, Path]],
    out_reference_dataset_csv: Union[str, Path],
    reference_column_name: str,
):
    """
    This function removes overlap between reaction datasets, a reference dataset
    is pruned by all reactions whose product is also present in the given list of supplementary
    datasets. Note: only the reference dataset is pruned.

        reference_dataset_csv: The dataset to prune
        input_files_csv: the input datasets to use for the pruning
        out_reference_dataset_csv: the otput file
        reference_column_name: the column name for the reactions (same for all datasets)
    """
    ref_df = pd.read_csv(reference_dataset_csv)
    logger.info(f"Reference dataset: {reference_dataset_csv}")
    logger.info(len(ref_df))
    ref_df["product"] = ref_df[reference_column_name].apply(lambda x: x.split(">>")[-1])
    reference_product_set = set(ref_df["product"].tolist())
    product_sets = []
    for f in input_files_csv:
        df = pd.read_csv(f)
        logger.info(f"Dataset: {f}")
        logger.info(len(df))
        df["product"] = df[reference_column_name].apply(lambda x: x.split(">>")[-1])
        product_sets.append(set(df["product"].tolist()))

    products_to_remove = []
    for s in product_sets:
        products_to_remove.extend(list(reference_product_set & s))
    logger.info(f"Number of products to remove: {len(products_to_remove)}")
    ref_df = ref_df[~ref_df["product"].isin(products_to_remove)]
    logger.info(f"Number of reactions left: {len(ref_df)}")
    ref_df.to_csv(out_reference_dataset_csv, index=False)
    logger.info(f"Dataset saved into {out_reference_dataset_csv}")


@click.command()
@click.argument("reference_dataset_csv", required=True)
@click.option("--input_files_csv", "-f", multiple=True, required=True)
@click.option("--reference_column_name", "-col", type=str, default="rxn")
def main(reference_dataset_csv: str, input_files_csv: List[str], reference_column_name: str):
    setup_console_logger()
    out_reference_dataset_csv = (
        Path(reference_dataset_csv).parent / f"{Path(reference_dataset_csv).name}.nooverlap"
    )
    remove_overlap(
        reference_dataset_csv,
        input_files_csv,
        out_reference_dataset_csv,
        reference_column_name,
    )


if __name__ == "__main__":
    main()
