import logging
from pathlib import Path

import click
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)


@click.command()
@click.option("--file1", type=str, required=True)
@click.option("--file2", type=str, required=True)
@click.option("--output_file", type=str, required=True)
def main(file1: str, file2: str, output_file: str):
    setup_console_logger()

    file1_path = Path(file1)
    file2_path = Path(file2)
    output_path = Path(output_file)

    logger.info(f"Output path for the dataset conversion: {str(output_path)}")

    merged_lines = []
    with open(file1_path, "r") as f, open(file2_path, "r") as g:
        for src, tgt in zip(f, g):
            merged_lines.append(src.strip())
            merged_lines.append(tgt.strip())

    # merged_lines = [json.dumps(l) for l in merged_lines]
    with open(output_path, "w") as f:
        f.write("\n".join(merged_lines))

    logger.info("Datasets merged successfully.")


if __name__ == "__main__":
    main()
