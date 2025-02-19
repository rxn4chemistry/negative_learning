import logging
from ast import literal_eval
from pathlib import Path
from typing import Union

import click
import pandas as pd
from rxn.chemutils.reaction_equation import ReactionEquation, rxn_standardization
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.data_generation.negative_reactions_generator import (
    PosNegRxnGenerator,
    RegioSQMdatum,
)
from rxn_negative_learning.scripts.extract_regiosqm_dataset import OUTPUT_FILE, load_regiosqm_data

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

INPUT_CSV = OUTPUT_FILE
OUTPUT_FILE_NAME = "regiosqm.csv"


def generate_regiosqm_negatives(
    input_csv: Union[str, Path], output_csv: Union[str, Path], keep_no_negatives: bool
):
    """
    Script to separate positives and negatives for the regiosqm dataset

    input_csv: input csv file, The columns 'name', 'main_reactant', 'reaction_centers', 'halo_reactants'
        contain respectively the name of the main reactant, the main reactant smiles, a list of reaction centers ids
        and a list of halogenated reactants.
    output_csv: where the splits will be stored in two different csv files
    keep_no_negatives: if True will keep the positives even if no negative exists
    """

    df = pd.read_csv(input_csv)
    df["reaction_centers"] = df["reaction_centers"].apply(lambda x: literal_eval(x))
    df["halo_reactants"] = df["halo_reactants"].apply(lambda x: literal_eval(x))

    logger.info(f"Sample of recation centers: {df['reaction_centers'].values[0:1]}")
    logger.info(f"Sample of halo reactants: {df['halo_reactants'].values[0:1]}")

    full_reactions = []
    reactions_ids = []
    scores = []
    idx = 0
    for index, (name, main_reactant, reaction_centers, halo_reactants) in df.iterrows():
        pos_neg_generator = PosNegRxnGenerator(
            RegioSQMdatum(name, main_reactant, reaction_centers, halo_reactants)
        )
        pos_reactions, pos_targets, neg_targets = pos_neg_generator()
        assert len(pos_targets) == len(neg_targets) == len(pos_targets)

        for pos_rxn, pos_tgt, neg_tgts in zip(pos_reactions, pos_targets, neg_targets):
            if neg_tgts is not None:
                full_reactions.append(f"{pos_rxn}>>{pos_tgt}")
                scores.append(1)
                reactions_ids.append(idx)
                for neg_tgt in neg_tgts:
                    full_reactions.append(f"{pos_rxn}>>{neg_tgt}")
                    scores.append(0)
                    reactions_ids.append(idx)
            elif keep_no_negatives:
                full_reactions.append(f"{pos_rxn}>>{pos_tgt}")
                scores.append(1)
                reactions_ids.append(idx)
            idx += 1
    output_df = pd.DataFrame({"rxn": full_reactions, "score": scores, "idx": reactions_ids})
    output_df["rxn"] = output_df["rxn"].apply(
        lambda x: rxn_standardization(ReactionEquation.from_string(x)).to_string()
    )
    print(output_df.loc[output_df.rxn == "COc1ccc(Br)cn1>>COc1ccc(Br)cn1"])
    logger.info(f"Number of positives: {len(output_df.score.loc[output_df.score == 1])}")
    logger.info(f"Number of negatives: {len(output_df.score.loc[output_df.score == 0])}")
    logger.info(output_df.head())

    output_df.to_csv(output_csv, index=False)


@click.command()
@click.argument("output_folder", type=str, required=True)
@click.option(
    "--keep_no_negatives",
    type=bool,
    default=False,
    is_flag=True,
    help="Keep reactions for which we were not able to generate a negative.",
)
def main(output_folder: str, keep_no_negatives: bool):
    setup_console_logger()
    output_dir = Path(output_folder)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Hello")
        # if any(Path(output_dir).iterdir()):
        #     raise FileExistsError("The output directory already exists and is not empty")
    output_csv = output_dir / OUTPUT_FILE_NAME

    logger.info("***DATA EXTRACTION***")
    load_regiosqm_data()
    logger.info("***DATA FORMATTING AND NEGATIVES GENERATION***")
    generate_regiosqm_negatives(INPUT_CSV, output_csv, keep_no_negatives)
    logger.info("***DONE!***")


if __name__ == "__main__":
    main()
