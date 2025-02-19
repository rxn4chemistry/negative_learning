import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd

from rxn_negative_learning.utils.smiles_utils import (
    get_precursors_from_smiles,
    get_product_from_smiles,
)


def convert_df_to_jsonl_list(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    other_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    # save a dictionary with all negative targets available for each positive one
    if "score" in df.keys() and "idx" in df.keys():
        neg_dict = {k: [] for k in set(df["idx"].values)}
        pos_dict = {k: "" for k in set(df["idx"].values)}
        for index, row in df.iterrows():
            if row["score"] == 0:
                neg_dict[row["idx"]].append(row[target_col])
            if row["score"] == 1:
                pos_dict[row["idx"]] = row[target_col]
    else:
        neg_dict = None
        pos_dict = None

    converted_lines = []
    for index, row in df.iterrows():
        if source_col in df.keys() and target_col in df.keys():
            d = {"source": row[source_col], "target": row[target_col]}
        else:
            raise ValueError("Provide correct names for columns in dataframe.")
        if other_cols is not None:
            for col in other_cols:
                if col in df.keys():
                    d[col] = row[col]
            if neg_dict is not None and row["score"] == 1:
                d["opposite_targets"] = neg_dict[row["idx"]]
            if pos_dict is not None and row["score"] == 0:
                d["opposite_targets"] = pos_dict[row["idx"]]

        converted_lines.append(d)
    return converted_lines


class NegativeSelection(Enum):
    random1 = "random1"
    priority1 = "priority1"
    all = "all"


@click.command()
@click.argument("input_file_csv", type=str, required=True)
@click.argument("output_path", type=str, required=True)
@click.option("--split_columns", "-scol", multiple=True, type=str)
@click.option("--reaction_column_name", "-col", type=str, default="rxn")
@click.option("--negative_selection", "-n", type=str, default="all", help="Currently unused")
@click.option("--randomize", is_flag=True, default=False)
def main(
    input_file_csv: str,
    output_path: str,
    split_columns: Tuple[str],
    reaction_column_name: str,
    negative_selection: str,
    randomize: bool,
):
    """
    This script generates a folder with files for each split present in the dataframe
    """

    df = pd.read_csv(input_file_csv)
    print(df.head())

    # Check that the negative selection method exists
    try:
        negative_sel = NegativeSelection(negative_selection)
    except ValueError:
        raise ValueError(f"{negative_selection} is not a valid negative selection method")

    if randomize:
        df = df.sample(n=len(df), random_state=42)
    print(df.head())

    for col in split_columns:
        print(f"Processing the column '{col}'")
        train = df.loc[df[col] == "train"]
        valid = df.loc[df[col] == "valid"]
        train_with_valid = df.loc[df[col] != "test"]
        test = df.loc[df[col] == "test"]
        rtrain = df.loc[df[col] == "rtrain"]

        columns_selection = [reaction_column_name, "score", "idx"]

        new_output_path = (
            Path(output_path) / "decreasingpos" / col / negative_sel.name
            if "_k" in col
            else (
                Path(output_path) / "scorersep" / col / negative_sel.name
                if "model" in df.keys()
                else Path(output_path) / col / negative_sel.name
            )
        )
        if new_output_path.exists():
            print(f"Rewriting the directory: {new_output_path}")
        new_output_path.mkdir(parents=True, exist_ok=True)

        train = train[columns_selection]
        valid = valid[columns_selection]
        train_with_valid = train_with_valid[columns_selection]
        test = test[columns_selection]
        rtrain = rtrain[columns_selection]

        for label, split_df in zip(
            ["train", "valid", "train-with-valid", "test", "rtrain"],
            [train, valid, train_with_valid, test, rtrain],
        ):
            # NOTE: rtrain does not contain the negative targets
            if label == "rtrain" and len(split_df) == 0:
                break
            # Save jsonl files
            split_df["precursors"] = split_df[reaction_column_name].apply(
                lambda x: get_precursors_from_smiles(x)
            )
            split_df["product"] = split_df[reaction_column_name].apply(
                lambda x: get_product_from_smiles(x)
            )
            converted_df = convert_df_to_jsonl_list(
                split_df,
                source_col="precursors",
                target_col="product",
                other_cols=["score", "idx"],
            )
            # Files with positive and negative reactions
            dumped_df = [json.dumps(line) for line in converted_df]
            output_jsonl = new_output_path / f"data-{label}-all.jsonl"
            with open(output_jsonl, "w") as f:
                f.write("\n".join(dumped_df))

            # Files with only positive reactions
            converted_pos_df = [line for line in converted_df if line["score"] == 1]
            dumped_df = [json.dumps(line) for line in converted_pos_df]
            output_jsonl = new_output_path / f"data-{label}-pos.jsonl"
            with open(output_jsonl, "w") as f:
                f.write("\n".join(dumped_df))

            # Files with only negative reactions
            converted_neg_df = [line for line in converted_df if line["score"] == 0]
            dumped_df = [json.dumps(line) for line in converted_neg_df]
            output_jsonl = new_output_path / f"data-{label}-neg.jsonl"
            with open(output_jsonl, "w") as f:
                f.write("\n".join(dumped_df))


if __name__ == "__main__":
    main()
