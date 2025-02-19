import logging
import random
from math import ceil, floor
from pathlib import Path
from typing import Optional, Tuple

import click
import pandas as pd
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.data_generation.data_splitter import (
    DataSplitter,
    NotSuitableParametersError,
    SplittingMethod,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def generate_multiple_splits(
    input_file_csv: str,
    output_file_csv: str,
    reaction_column_name: str,
    idx_column_name: str,
    split_ratio: float,
    splitting_method: str,
    seed: Tuple[int],
    validation_set: bool,
    remove_duplicate_columns: bool,
    scorer_dataset_split_ratio: Optional[float] = None,
):
    """
    This function is used to generate multiple splits (eg. different seed) of the same dataset
        according to a chosen type of splitting
    Parameters
    ----------
    input_file_csv: the input csv file containing a column with reaction smiles
    output_file_csv: the name of the output csv file
    reaction_column_name: the name of the column containing the reaction smiles
    idx_column_name: the name of the column containing the id of the refence compound,
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
    remove_duplicate_columns: if set to True, just one instance of columns
        with similar splitting is kept (this can be useful for small datasets)
    scorer_dataset_split_ratio: if a float is given, a ratio of the dataset is used to train/test/validate
        the scorer. In addition, the training set of the scorer will always be part of the training of
        the rl models.

    """

    # Import input file
    df = pd.read_csv(input_file_csv)

    # Check that the splitting method exists
    try:
        splitting_met = SplittingMethod(splitting_method)
    except ValueError as e:
        logger.exception(e)
        raise

    # Remove the scorer df if asked
    df_scorer = None
    if scorer_dataset_split_ratio is not None:
        random.seed(10)
        # The hash is the idx of the reaction
        df["hash"] = df[idx_column_name]
        list_of_unique_ids = sorted([key for key, value in df["hash"].value_counts().items()])
        number_of_unique_ids = len(list_of_unique_ids)

        list_of_scorer_ids = random.sample(
            list_of_unique_ids,
            k=floor(number_of_unique_ids * scorer_dataset_split_ratio),
        )
        list_of_unique_ids = sorted(list_of_scorer_ids)
        number_of_unique_ids = len(list_of_unique_ids)
        list_of_scorer_test_ids = random.sample(
            list_of_unique_ids, k=floor(number_of_unique_ids * split_ratio)
        )

        list_of_unique_ids = sorted(list(set(list_of_unique_ids) - set(list_of_scorer_test_ids)))

        if validation_set:
            list_of_scorer_valid_ids = random.sample(
                list_of_unique_ids, k=ceil(number_of_unique_ids * split_ratio)
            )
            list_of_unique_ids = list(set(list_of_unique_ids) - set(list_of_scorer_valid_ids))

        df["model"] = df.hash.apply(
            lambda x: "ctest"
            if x in list_of_scorer_test_ids
            else "ctrain"
            if x in list_of_unique_ids
            else "cvalid"
            if x in list_of_scorer_valid_ids
            else "rl"
        )
        logger.info("Stats for scorer vs rl datasets.")
        print(df["model"].value_counts(normalize=True))
        logger.info("Numbers for scorer vs rl datasets.")
        print(df["model"].value_counts())
        df_scorer = df.loc[df.model != "rl"].reset_index(drop=True)
        df_scorer = df_scorer.drop(["hash"], axis=1)
        df = df.loc[df.model == "rl"].reset_index(drop=True)

        scorer_output_file_csv = (
            Path(output_file_csv).parent
            / f"scorer.rlratio{scorer_dataset_split_ratio}.ratio{split_ratio}.csv"
        )
        df_scorer.to_csv(scorer_output_file_csv, index=False)

    for s in seed:
        try:
            DataSplitter.split(
                df=df,
                reaction_column_name=reaction_column_name,
                idx_column_name=idx_column_name,
                splitting_method=splitting_met,
                split_ratio=split_ratio,
                seed=s,
                validation_set=validation_set,
            )

        except NotSuitableParametersError:
            df = df.drop([f"sratio{split_ratio}_{splitting_method}_seed{seed}"], axis=1)

    if remove_duplicate_columns:
        df = df.loc[:, ~df.T.duplicated(keep="first")]

    print(df.head())

    if df_scorer is not None:
        df = pd.concat([df, df_scorer.loc[df_scorer["model"] == "ctrain"]]).fillna("train")
        logger.info("New Stats after scorer training set inclusion")
        for s in seed:
            logger.info(f"seed {s} with method '{splitting_method}'.")
            print(
                df[f"sratio{split_ratio}_{splitting_method}_seed{s}"].value_counts(normalize=True)
            )
            print(df[f"sratio{split_ratio}_{splitting_method}_seed{s}"].value_counts())

    df.to_csv(output_file_csv, index=False)


@click.command()
@click.argument("input_file_csv", type=str, required=True)
@click.argument("output_file_csv", type=str, required=True)
@click.option("--reaction_column_name", "-col", type=str, default="rxn")
@click.option("--idx_column_name", "-col", type=str, default="idx")
@click.option("--split_ratio", "-sr", type=float, default=0.01)
@click.option("--splitting_method", "-sm", type=str, default="random")
@click.option("--seed", multiple=True, type=int)
@click.option("--validation_set", is_flag=True, default=False)
@click.option("--remove_duplicate_columns", is_flag=True, default=False)
@click.option("--scorer_dataset_split_ratio", type=float, default=None)
def main(
    input_file_csv: str,
    output_file_csv: str,
    reaction_column_name: str,
    idx_column_name: str,
    split_ratio: float,
    splitting_method: str,
    seed: Tuple[int],
    validation_set: bool,
    remove_duplicate_columns: bool,
    scorer_dataset_split_ratio: Optional[float],
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

    generate_multiple_splits(
        input_file_csv,
        output_file_csv,
        reaction_column_name,
        idx_column_name,
        split_ratio,
        splitting_method,
        seed,
        validation_set,
        remove_duplicate_columns,
        scorer_dataset_split_ratio,
    )


if __name__ == "__main__":
    main()
