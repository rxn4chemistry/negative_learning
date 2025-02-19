import json
import logging
from pathlib import Path

import click
import pandas as pd
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)


@click.command()
@click.option("--input_file_jsonl", type=str, required=True)
@click.option("--output_file", type=str, default=None)
@click.option("--randomize", type=bool, is_flag=True, default=False)
def main(input_file_jsonl: str, output_file: str, randomize=bool):
    setup_console_logger()

    source_path = Path(input_file_jsonl)

    if output_file is None:
        output_path = source_path.parent / "dataset_for_scorer.csv"
    else:
        output_path = Path(output_file)

    logger.info(f"Output path for the dataset conversion: {str(output_path)}")

    rxn = []
    rxn_class = []
    with open(source_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
        for elem in data:
            rxn.append(f"{elem['source']}>>{elem['target']}")
            rxn_class.append(elem["score"])

    df = pd.DataFrame({"rxn": rxn, "rxn_class": rxn_class})
    print(df.head())
    if randomize:
        df = df.sample(n=len(df), random_state=42)
    print(f"Number of positives: {len(df.loc[df['rxn_class'] == 1])}")
    print(f"Number of negatives: {len(df.loc[df['rxn_class']] == 0)}")

    df.to_csv(output_path, index=False)

    logger.info("Dataset converted successfully.")


if __name__ == "__main__":
    main()
