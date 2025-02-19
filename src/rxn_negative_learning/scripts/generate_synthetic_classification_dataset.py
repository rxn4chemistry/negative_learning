import logging
import random
from pathlib import Path

import click
import pandas as pd
from rdkit import Chem
from rxn.chemutils.tokenization import detokenize_smiles
from rxn.utilities.containers import chunker
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.metrics import get_sequence_multiplier
from rxn_negative_learning.utils.smiles_utils import lazy_canonicalize_smiles

logger = logging.getLogger(__name__)


@click.command()
@click.option("--predictions_file_txt", type=str, required=True)
@click.option("--targets_file_txt", type=str, required=True)
@click.option("--strategy", type=click.Choice(["all", "one"]), default="one")
@click.option("--randomize", type=bool, is_flag=True, default=False)
def main(predictions_file_txt: str, targets_file_txt: str, strategy: str, randomize: bool):
    setup_console_logger()

    predictions_path = Path(predictions_file_txt)
    targets_path = Path(targets_file_txt)

    output_path_train = predictions_path.parent / "dataset_for_reward_model_train.csv"
    output_path_valid = predictions_path.parent / "dataset_for_reward_model_valid.csv"

    logger.info(
        f"Output paths for the dataset conversion: {str(output_path_train), str(output_path_valid)}"
    )
    logger.info(f"Using strategy: {strategy}")
    logger.info("Assuming the files of targets is canonical ...")
    logger.info("Canonicalizing predictions ...")

    with open(predictions_path, "r") as f:
        predictions = [lazy_canonicalize_smiles(detokenize_smiles(line.strip())) for line in f]
        logger.info(f"Number of predictions: {len(predictions)}")
    with open(targets_path, "r") as f:
        tgt_rxn = [line.strip() for line in f]
        inputs = [elem.split(">>")[0] for elem in tgt_rxn]
        targets = [elem.split(">>")[-1] for elem in tgt_rxn]
        logger.info(f"Number of targets: {len(targets)}")
        logger.info(f"Number of inputs: {len(inputs)}")

    try:
        print("Inputs")
        [Chem.MolToSmiles(Chem.MolFromSmiles(inp)) for inp in inputs]
        print("Targets")
        [Chem.MolToSmiles(Chem.MolFromSmiles(inp)) for inp in targets]
        print("Predictions")
        [Chem.MolToSmiles(Chem.MolFromSmiles(inp)) for inp in predictions]
    except:
        raise
    multiplier = get_sequence_multiplier(ground_truth=targets, predictions=predictions)
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    # Initializing with the positive dataset
    reactions = [f"{inputs[i]}>>{targets[i]}" for i in range(len(targets))]
    scores = [1 for _ in range(len(targets))]
    indexes = [i for i in range(len(targets))]

    # Adding the negative dataset
    for idx, inp, gt, predictions in zip(indexes, inputs, targets, prediction_chunks):
        for i in range(multiplier):
            if predictions[i] != gt and predictions[i] != "":
                reactions.append(f"{inp}>>{predictions[i]}")
                scores.append(0)
                indexes.append(idx)
                if strategy == "one":
                    break

    df = pd.DataFrame({"rxn": reactions, "rxn_class": scores, "idx": indexes})
    logger.info(f"Number of positive samples: {len(df.loc[df['rxn_class'] == 1])}")
    logger.info(f"Number of negative samples: {len(df.loc[df['rxn_class'] == 0])}")
    logger.info(f"Sample: {df.loc[df['idx'] == 20].values}")

    logger.info(df.head())

    # Sampling is performed keeping a positive and a corresponding negative together
    indexes = list(set(indexes))
    train_ids = random.sample(indexes, k=int(len(indexes) * 0.8))
    df["train"] = df.apply(lambda x: x["idx"] in train_ids, axis=1)
    train = df.loc[df["train"]]
    valid = df.loc[~df["train"]]
    train = train[["rxn", "rxn_class"]]
    valid = valid[["rxn", "rxn_class"]]

    if randomize:
        train = train.sample(n=len(train), random_state=42)
        valid = valid.sample(n=len(valid), random_state=42)

    for name, split in zip(["train", "valid"], [train, valid]):
        logger.info(
            f"Number of positive samples in {name}: {len(split.loc[split['rxn_class'] == 1])}"
        )
        logger.info(
            f"Number of negative samples in {name}: {len(split.loc[split['rxn_class'] == 0])}"
        )

    train.to_csv(output_path_train, index=False)
    valid.to_csv(output_path_valid, index=False)

    logger.info("Dataset generated successfully.")


if __name__ == "__main__":
    main()
