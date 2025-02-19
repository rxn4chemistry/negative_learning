"""Testing routine."""

import logging
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.models.base_transformer.ccc import fix_infiniband, set_env
from rxn_negative_learning.models.base_transformer.pytorch_lightning.dataset import LitSmilesDataset
from rxn_negative_learning.models.base_transformer.pytorch_lightning.model import (
    LitVanillaTransformer,
)
from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.utils.smiles_utils import flatten

logger = logging.getLogger("__name__")


def main():
    # CCC Specific Code --------------------------------------------------------------------------------------------------------------------------
    fix_infiniband()
    set_env()
    # --------------------------------------------------------------------------------------------------------------------------------------------
    setup_console_logger()
    pl.seed_everything(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)  # type: ignore
    parser = LitVanillaTransformer.add_model_specific_args(parser)
    parser = LitSmilesDataset.add_dataset_specific_args(parser)
    parser.add_argument("--vocabulary", type=str, default="/Users/ray/Code/mit_data/")
    parser.add_argument("--ckpt", type=str, default="./test.ckpt")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--no_teacher_forcing", action="store_true")
    parser.add_argument("--num_predictions_per_sample", type=int, default=1)

    args = parser.parse_args()
    # ------------
    # data
    # ------------

    tokenizer = SmilesTokenizer(vars(args)["vocabulary"])
    smiles_dataset = LitSmilesDataset(dataset_args=vars(args), tokenizer=tokenizer)

    smiles_dataset.load()

    test_dataloader = smiles_dataset.test_dataloader()

    # ------------
    # model
    # ------------

    model = LitVanillaTransformer(model_args=vars(args), tokenizer=tokenizer)

    # ------------
    # testing
    # ------------

    trainer = pl.Trainer.from_argparse_args(args)

    # testing the model automatically selecting the checkpoint to be loaded
    # predictions = trainer.test(model, dataloaders=test_dataloader, ckpt_path=vars(args)["ckpt"])
    outputs = trainer.predict(model, dataloaders=test_dataloader, ckpt_path=vars(args)["ckpt"])
    LD_to_DL_outputs = {k: flatten([dic[k] for dic in outputs]) for k in outputs[0]}
    logger.info(f"Outputs: {outputs}")
    predictions, scores = LD_to_DL_outputs["predictions"], LD_to_DL_outputs["scores"]
    logger.info(f"Number of predictions: {len(outputs)}")
    logger.info(f"Number of predictions * beams: {len(predictions)}")
    logger.info(f"Number of scores * beams: {len(scores)}")

    if vars(args)["output_file"] is None:
        output_dir = Path(vars(args)["ckpt"]).parent / "predict"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "predictions.txt"
    else:
        output_dir = Path(vars(args)["output_file"]).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(vars(args)["output_file"])

    logger.info(f"Predictions will be saved to: {output_file}")
    with open(output_file, "w") as f:
        f.write("\n".join(predictions))
    with open(f"{output_file}.scores", "w") as f:
        f.write("\n".join([str(s.item()) for s in scores]))
