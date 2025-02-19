import json
import logging
from pathlib import Path

import click
from rxn.chemutils.tokenization import detokenize_smiles
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)


@click.command()
@click.option("--source_file", type=str, required=True)
@click.option("--target_file", type=str, required=True)
@click.option("--score", type=int, required=True)
@click.option("--output_file", type=str, default=None)
def main(source_file: str, target_file: str, score: int, output_file: str):
    setup_console_logger()

    source_path = Path(source_file)
    target_path = Path(target_file)

    if output_file is None:
        output_path = source_path.parent / "dataset.jsonl"
    else:
        output_path = Path(output_file)

    logger.info(f"Output path for the dataset conversion: {str(output_path)}")

    converted_lines = []
    with open(source_path, "r") as f, open(target_path, "r") as g:
        for src, tgt in zip(f, g):
            converted_lines.append({
                "source": detokenize_smiles(src.strip()),
                "target": detokenize_smiles(tgt.strip()),
                "score": score,
            })

    converted_lines = [json.dumps(line) for line in converted_lines]
    with open(output_path, "w") as f:
        f.write("\n".join(converted_lines))

    logger.info("Dataset converted successfully.")


if __name__ == "__main__":
    main()
