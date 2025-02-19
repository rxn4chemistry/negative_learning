import json
import logging
from pathlib import Path

import click
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)


@click.command()
@click.option("--jsonl_file", type=str, required=True)
@click.option("--output_file", type=str, default=None)
def main(jsonl_file: str, output_file: str):
    setup_console_logger()

    source_path = Path(jsonl_file)

    if output_file is None:
        output_path = source_path.parent / "rxn-dataset.txt"
    else:
        output_path = Path(output_file)

    logger.info(f"Output path for the dataset conversion: {str(output_path)}")

    txt_lines = []
    with open(source_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    for elem in data:
        txt_lines.append(f"{elem['source']}>>{elem['target']}")

    with open(output_path, "w") as f:
        f.write("\n".join(txt_lines))

    logger.info("Dataset converted successfully.")


if __name__ == "__main__":
    main()
