import logging
from pathlib import Path

import click
import pandas as pd
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.data_generation.data_splitter import DataSplitter, SplittingMethod

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generate_decreasing_positive_splits(
    input_file_csv: str,
    output_file_csv: str,
    reaction_column_name: str,
    idx_column_name: str,
    split_ratio: float,
    splitting_method: str,
    seed: int,
    validation_set: bool,
):
    """
    This function is used to generate splits (eg. different seed) of the same dataset
        that have a decreasing number of positives, but keep the negatives
    Parameters
    ----------
    input_file_csv: the input csv file containing a column with reaction smiles
    output_file_csv: the name of the output csv file
    reaction_column_name: the name of the column containing the reaction smiles
    idx_column_name: the name of the column containing the id of the reference compound,
        this allows positives and negatives to stay in the same split
    split_ratio: the splitting ratio between train/test or train/test/valid
            e.g. split_ratio=0.1 means 10%test and 90%train OR
            10%test and 10%valid and 80%train if the validation_set flag is set to True
    splitting_method: the type of splitting method. Available ones are
                SplittingMethod.random = reactions are splitted randomly
                    This is NOT STABLE for a new version of the same dataset
                SplittingMethod.product = the unique products are splitted randomly:
                    reactions with the same product end up in the same split. Use this
                    instead of "product_hash" for a dataset with few products.
                    This is NOT STABLE for a new version of the same dataset.
                SplittingMethod.product_hash = the unique products are hashed and
                    then splitted according to the hash value.
                    This is STABLE for a new version of the same dataset
                SplittingMethod.product_tanimoto = not yet implemented
    seed: the seed to use to generate the dataset. Changing seed will change the splitting
    validation_set: whether or not to generate a validation set
    remove_duplicate_columns: if set to True, repeated equal splits are removed

    """

    # Import input file
    df = pd.read_csv(input_file_csv)

    # Check that the splitting method exists
    try:
        splitting_met = SplittingMethod(splitting_method)
    except ValueError as e:
        logger.exception(e)
        raise

    # first round of splitting
    DataSplitter.split(
        df=df,
        reaction_column_name=reaction_column_name,
        idx_column_name=idx_column_name,
        splitting_method=splitting_met,
        split_ratio=split_ratio,
        seed=seed,
        validation_set=validation_set,
    )

    # increasingly remove positives until no one is left
    number_of_positives_left = len(
        df.loc[
            (df[f"sratio{split_ratio}_{splitting_met.value}_seed{seed}"] == "train")
            & (df["score"] == 1)
        ]
    )
    n_to_remove = round(number_of_positives_left * split_ratio)
    k = 0
    logger.info(f"Number of positives in fold {k}: {number_of_positives_left}")
    while number_of_positives_left > 0:
        k += 1
        active_column = (
            f"sratio{split_ratio}_{splitting_met.value}_seed{seed}_k{k - 1}"
            if k > 1
            else f"sratio{split_ratio}_{splitting_met.value}_seed{seed}"
        )
        # remove positive examples from the training set with a predefined split ratio
        df_to_remove = (
            df.loc[(df[active_column] == "train") & (df["score"] == 1)].sample(
                n=n_to_remove, random_state=seed
            )
            if number_of_positives_left > n_to_remove
            else df.loc[(df[active_column] == "train") & (df["score"] == 1)]
        )
        df[f"sratio{split_ratio}_{splitting_met.value}_seed{seed}_k{k}"] = df[active_column]
        df.loc[
            df_to_remove.index,
            f"sratio{split_ratio}_{splitting_met.value}_seed{seed}_k{k}",
        ] = "rtrain"
        number_of_positives_left = len(
            df.loc[
                (df[f"sratio{split_ratio}_{splitting_met.value}_seed{seed}_k{k}"] == "train")
                & (df["score"] == 1)
            ]
        )
        logger.info(f"Number of positives in fold {k}: {number_of_positives_left}")

    df = df.rename(
        {
            f"sratio{split_ratio}_{splitting_met.value}_seed{seed}": f"sratio{split_ratio}_{splitting_met.value}_seed{seed}_k0"
        },
        axis=1,
    )
    df.to_csv(output_file_csv, index=False)
    logger.info(f"Wrote results to file: {output_file_csv}")


@click.command()
@click.argument("input_file_csv", type=str, required=True)
@click.argument("output_file_csv", type=str, required=True)
@click.option("--reaction_column_name", "-col", type=str, default="rxn")
@click.option("--idx_column_name", "-col", type=str, default="idx")
@click.option("--split_ratio", "-sr", type=float, default=0.1)
@click.option("--splitting_method", "-sm", type=str, default="random")
@click.option("--seed", type=int, default=42)
@click.option("--validation_set", is_flag=True, default=False)
def main(
    input_file_csv: str,
    output_file_csv: str,
    reaction_column_name: str,
    idx_column_name: str,
    split_ratio: float,
    splitting_method: str,
    seed: int,
    validation_set: bool,
):
    setup_console_logger()
    output_csv = Path(output_file_csv)
    output_dir = output_csv.parent
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        logger.info("The output directory already exists.")
        if any(Path(output_dir).iterdir()):
            logger.info("The output directory is not empty.")

    generate_decreasing_positive_splits(
        input_file_csv,
        output_file_csv,
        reaction_column_name,
        idx_column_name,
        split_ratio,
        splitting_method,
        seed,
        validation_set,
    )


if __name__ == "__main__":
    main()
