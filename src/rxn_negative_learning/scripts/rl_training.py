"""Training and validation routine."""

import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.data_generation.data_standardizer import SMILESDataStandardizer
from rxn_negative_learning.models.base_transformer.ccc import fix_infiniband, set_env
from rxn_negative_learning.models.base_transformer.pytorch_lightning.dataset import LitSmilesDataset
from rxn_negative_learning.models.base_transformer.pytorch_lightning.model import (
    LitVanillaTransformer,
)
from rxn_negative_learning.models.custom_callbacks import Grad2NormCallback, MaxGradCallback
from rxn_negative_learning.models.rl_transformer.reinforce_lightning import (
    ReinforceLitVanillaTransformer,
)
from rxn_negative_learning.models.scorers.ideal_scorer import IdealScorer
from rxn_negative_learning.models.scorers.levenstein_ideal_scorer import LevenshteinIdealScorer
from rxn_negative_learning.models.scorers.scorer_base import RXNNegScorerBase
from rxn_negative_learning.models.scorers.svm_scorer import SVMScorer
from rxn_negative_learning.models.scorers.tanimoto_ideal_scorer import TanimotoIdealScorer
from rxn_negative_learning.models.tokenization import SmilesTokenizer


def main():
    # CCC Specific Code -----------------------------------------------------------------------------------------------
    fix_infiniband()
    set_env()
    # -----------------------------------------------------------------------------------------------------------------
    setup_console_logger()
    pl.seed_everything(42, workers=True)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)  # type: ignore
    parser = LitVanillaTransformer.add_model_specific_args(parser)
    parser = LitSmilesDataset.add_dataset_specific_args(parser)
    parser.add_argument("--vocabulary", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scorer_path", type=str, required=True)
    parser.add_argument("--no_teacher_forcing", action="store_true")
    parser.add_argument("--additional_samples_file", type=str, default=None)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--baseline_targets_file", type=str, default=None)  # **
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--filename", type=str, default="{epoch}-{step}-{accuracy_pos_valid:.2f}")
    parser.add_argument("--every_n_train_steps", type=int, default=100)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--monitor", type=str, default="accuracy_pos_valid")
    parser.add_argument("--mode", type=str, default="max")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="linear")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--save_validation_predictions", action="store_true")
    parser.add_argument("--save_train_predictions", action="store_true")
    parser.add_argument("--with_baseline", type=str, default="sigmoid")
    parser.add_argument("--baseline_model_lr", type=float, default=5e-6)
    parser.add_argument("--baseline_model_dropout", type=float, default=0.1)
    parser.add_argument("--baseline_model_oversampling_threshold", type=float, default=0.1)
    parser.add_argument("--baseline_model_weight_decay", type=float, default=1e-3)
    parser.add_argument("--randomic", action="store_true")  # Randomic baseline
    parser.add_argument("--regularization", type=str, default="kl-reverse")
    parser.add_argument("--regularization_beta", type=float, default=10)
    parser.add_argument("--loss_regularization_reference", type=float, default=10)
    parser.add_argument("--K_beta", type=float, default=1e-1)
    parser.add_argument("--update_lookup_table_every_n_epochs", type=Optional[int], default=None)
    parser.add_argument("--dynamic_beta", type=str, default="clipping")
    parser.add_argument("--update_regularization_model_every_step", type=Optional[int], default=None)
    parser.add_argument("--baseline_batch_size", type=int, default=64)
    parser.add_argument("--linear_lr_total_iters", type=int, default=10000)
    parser.add_argument("--linear_lr_end_factor", type=float, default=0.0)
    parser.add_argument("--cosine_lr_t_max", type=int, default=1000)
    parser.add_argument("--polynomial_lr_power", type=float, default=0.5)
    args = parser.parse_args()
    model_args = vars(args)

    # ------------
    # data
    # ------------
    tokenizer = SmilesTokenizer(model_args["vocabulary"])
    smiles_dataset = LitSmilesDataset(dataset_args=model_args, tokenizer=tokenizer)

    smiles_dataset.load()
    positive_samples = []
    negative_samples = []
    if model_args["additional_samples_file"] is not None:
        with open(model_args["additional_samples_file"], "r") as f:
            data = [json.loads(line.strip()) for line in f]
            for sample in data:
                if sample["score"] == 1:
                    rxn = SMILESDataStandardizer().standardize(
                        f"{sample['source']}>>{sample['target']}"
                    )
                    positive_samples.append(rxn)
                elif sample["score"] == 0:
                    rxn = SMILESDataStandardizer().standardize(
                        f"{sample['source']}>>{sample['target']}"
                    )
                    negative_samples.append(rxn)
            print("Saved additionally provided samples")
        positive_samples = list(set(positive_samples))
        negative_samples = list(set(negative_samples))
        print(f"Saved unique positive samples, total: {len(positive_samples)}")
        print(f"Saved unique negative samples, total: {len(negative_samples)}")
    else:
        positive_samples = None
        negative_samples = None

    train_dataloader = smiles_dataset.train_dataloader()
    val_dataloader = smiles_dataset.val_dataloader()

    # ------------
    # model
    # ------------

    # Build scorer model

    IDEAL_SCORERS = {
        "basic": IdealScorer,
        "levenshtein": LevenshteinIdealScorer,
        "tanimoto": TanimotoIdealScorer,
    }

    scorer: RXNNegScorerBase

    if model_args["scorer_path"] in IDEAL_SCORERS.keys():
        print(f"Using the ideal scorer: {model_args['scorer_path']}")
        scorer = IDEAL_SCORERS[model_args["scorer_path"]](
            positive_reactions=positive_samples, negative_reactions=negative_samples
        )
    elif Path(model_args["scorer_path"]).is_dir():
        print(f"Using the scorer at: {model_args['scorer_path']}")
        scorer = SVMScorer(
            model_path=Path(model_args["scorer_path"]),
            positive_reactions=positive_samples,
            negative_reactions=negative_samples,
        )
    else:
        raise ValueError("No valid ideal scorer provided or path for SVM scorer!")

    # ------------
    # training
    # ------------

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_args["output_dir"],
        filename=model_args["filename"],  # "{epoch:02d}-{step}",
        every_n_train_steps=model_args["every_n_train_steps"],
        save_top_k=model_args["save_top_k"],
        monitor=model_args["monitor"],
        mode=model_args["mode"],
    )
    gradl2_callback = Grad2NormCallback()
    maxgrad_callback = MaxGradCallback()

    if model_args["output_dir"] is not None:
        logger = TensorBoardLogger(model_args["output_dir"], name="tensorboard_output")
    else:
        logger = True  # uses the default one, which is still TensorBoard

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, lr_monitor, gradl2_callback, maxgrad_callback],
        logger=logger,
        deterministic=True,
    )

    if model_args["resume"] and model_args["finetune"]:
        raise ValueError("Cannot use the flags '--finetune' and '--resume' together!")

    if model_args["resume"]:
        print(f"Resuming training from where it ended: {model_args['ckpt']}")
        model = ReinforceLitVanillaTransformer(
            model_args=model_args,
            tokenizer=tokenizer,
            scorer=scorer,
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=model_args["ckpt"],
        )
    elif model_args["finetune"]:
        print(f"Finetuning from the model: {model_args['ckpt']}")
        model = ReinforceLitVanillaTransformer.load_from_checkpoint(
            model_args["ckpt"], scorer=scorer, model_args=model_args, strict=False
        )
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        print("Training from scratch")
        model = ReinforceLitVanillaTransformer(
            model_args=model_args,
            tokenizer=tokenizer,
            scorer=scorer,
        )
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
