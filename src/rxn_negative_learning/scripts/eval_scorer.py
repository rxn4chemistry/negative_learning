import logging
from pathlib import Path

import click
from rxn.utilities.files import dump_list_to_file, load_list_from_file
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.models.scorers.model import RXNNegScorerModel
from rxn_negative_learning.models.tokenization import SmilesTokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--model_path",
    "-m",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to model training checkpoint",
)
@click.option(
    "--input_txt",
    "-i",
    type=click.Path(exists=True, file_okay=True, path_type=Path),
    required=True,
    help="Input file with list of reactions",
)
@click.option(
    "--output_txt",
    "-o",
    type=click.Path(exists=False, path_type=Path),
    required=True,
    help="Output file where either the fingerprints or the reaction classes are stored",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["classification", "fingerprints"], case_sensitive=True),
    required=True,
    help="The functioning mode: either `classification` or `fingerprints`",
)
@click.option(
    "--batch_size",
    type=int,
    default=16,
    help="The batch_size for the classification model",
)
def main(model_path: Path, input_txt: Path, output_txt: Path, mode: str, batch_size: int) -> None:
    setup_console_logger()

    rxns = load_list_from_file(input_txt)
    logger.info(f"Running predictions for {len(rxns)} reactions")
    logger.info("`classification` mode enabled!")
    tokenizer = SmilesTokenizer.from_pretrained(model_path)
    class_model = RXNNegScorerModel(
        str(model_path), tokenizer=tokenizer
    )  # , batch_size=batch_size)
    output = list(class_model.predict(rxns))
    # if mode == "fingerprints":
    #     logger.info("`fingerprints` mode enabled!")
    #     rxnfp_model = RXNFPModel(model_name_or_path=model_path, batch_size=batch_size)
    #     output = list(rxnfp_model.predict(rxns))

    dump_list_to_file(filename=output_txt, values=output)  # type: ignore
    logger.info(f"Predictions saved to: {output_txt}")


if __name__ == "__main__":
    main()
