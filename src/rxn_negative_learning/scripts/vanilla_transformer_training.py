"""Training and validation routine."""

from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.models.base_transformer.ccc import fix_infiniband, set_env
from rxn_negative_learning.models.base_transformer.pytorch_lightning.dataset import LitSmilesDataset
from rxn_negative_learning.models.base_transformer.pytorch_lightning.model import (
    LitVanillaTransformer,
)
from rxn_negative_learning.models.custom_callbacks import MaxGradCallback
from rxn_negative_learning.models.tokenization import SmilesTokenizer


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
    parser.add_argument("--vocabulary", type=str, default="./smiles-vocab.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--every_n_train_steps", type=int, default=None)
    parser.add_argument("--save_top_k", type=int, default=1)
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--mode", type=str, default="min")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr_scheduler", action="store_true")
    parser.add_argument("--lr_scheduler_type", type=str, default="exponential")
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--no_teacher_forcing", action="store_true")
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    tokenizer = SmilesTokenizer(vars(args)["vocabulary"])
    smiles_dataset = LitSmilesDataset(dataset_args=vars(args), tokenizer=tokenizer)

    smiles_dataset.load()

    train_dataloader = smiles_dataset.train_dataloader()
    val_dataloader = smiles_dataset.val_dataloader()

    # ------------
    # training
    # ------------

    model_checkpoint_args = vars(args)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_checkpoint_args["output_dir"],
        filename=model_checkpoint_args["filename"],  # "{epoch:02d}-{step}",
        every_n_train_steps=model_checkpoint_args["every_n_train_steps"],
        save_top_k=model_checkpoint_args["save_top_k"],
        monitor=model_checkpoint_args["monitor"],
        mode=model_checkpoint_args["mode"],
    )

    maxgrad_callback = MaxGradCallback()

    if model_checkpoint_args["output_dir"] is not None:
        logger = TensorBoardLogger(model_checkpoint_args["output_dir"], name="tensorboard_output")
    else:
        logger = True  # uses the default one, which is still TensorBoard

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, lr_monitor, maxgrad_callback], logger=logger
    )

    # ------------
    # model
    # ------------

    if model_checkpoint_args["resume"] and model_checkpoint_args["finetune"]:
        raise ValueError("Cannot use the flags '--finetune' and '--resume' together!")

    if model_checkpoint_args["resume"]:
        print(f"Resuming training from where it ended: {model_checkpoint_args['ckpt']}")
        model = LitVanillaTransformer(model_args=vars(args), tokenizer=tokenizer)
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=model_checkpoint_args["ckpt"],
        )
    elif model_checkpoint_args["finetune"]:
        print(f"Finetuning from the model: {model_checkpoint_args['ckpt']}")
        model = LitVanillaTransformer.load_from_checkpoint(
            model_checkpoint_args["ckpt"], model_args=vars(args)
        )
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    else:
        print("Training from scratch")
        model = LitVanillaTransformer(model_args=vars(args), tokenizer=tokenizer)
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
