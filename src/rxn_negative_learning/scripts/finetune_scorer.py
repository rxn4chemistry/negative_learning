import json
import logging
import random
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import pandas as pd
import torch
from datasets import load_metric
from rxn.utilities.logging import setup_console_logger
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.utils.smiles_utils import (
    oversample_reaction_minority_dataset,
    randomize_multiple_smiles_rotated,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

LABEL_FILENAME = "labels.txt"


def get_associated_max_len(checkpoint_dir: Path) -> int:
    """Get the max sequence length by looking in the config file associated with
    a model checkpoint."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"No config file found where expected: {config_path}")

    try:
        with open(config_path, "rt") as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f'Error when parsing "{config_path}": {e}')

    try:
        return config["max_position_embeddings"]
    except KeyError:
        raise RuntimeError(
            "Can't determine max sequence length: did not find "
            f'"max_position_embeddings" in "{config_path}"'
        )


class MultiLabelDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, max_len: int, augment: bool
    ):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index: int):
        text = str(self.text[index])
        text = " ".join(text.split())

        if self.augment and random.choice([True, False]):
            # TODO: check that this changes at every epoch
            text = randomize_multiple_smiles_rotated(text)

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(self.targets[index], dtype=torch.float),
        }


def finetune(
    pretrained_model: Path,
    tokenizer: PreTrainedTokenizer,
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_model_path: Path,
    epochs: int,
    batch_size: int,
    valid_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    resume_from_checkpoint: Optional[str] = None,
    rebalance: bool = False,
    smoothing: Optional[float] = None,
    weights: Optional[List[float]] = None,
    freeze_encoder: bool = False,
    augment: bool = False,
    logging_steps: int = 500,
    warmup_steps: int = 500,
    oversample: bool = False,
) -> None:
    """Finetune a language-based, BERT-type model."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    set_seed(42)

    logger.info("Getting labels and transforming data...")

    if rebalance and weights is not None:
        raise ValueError("Either give weights OR choose to rebalance!")

    unique_labels = np.unique(training_df[label_column].values)
    if rebalance:
        logger.info("Rebalancing the dataset ...")
        tot_samples = len(training_df)
        WEIGHTS = []
        for lab in unique_labels:
            num_samples = len(training_df.loc[training_df[label_column] == lab])
            logger.info(f"Ratio of label {lab} samples: {num_samples / tot_samples:.2f}")
            WEIGHTS.append((1 / num_samples) * (tot_samples / 2.0))
        if smoothing is not None:
            bigger_weight = np.argmax(WEIGHTS)
            WEIGHTS[bigger_weight] = WEIGHTS[bigger_weight] * smoothing
    else:
        if weights is not None:
            WEIGHTS = weights
        else:
            WEIGHTS = [1.0 for _ in range(len(unique_labels))]
    logger.info(f"Computed weights: {WEIGHTS}")

    # lb = LabelBinarizer()
    logger.info("Assuming a regression ...")
    training_df["text"] = training_df[text_column]
    training_df["labels"] = training_df[label_column]
    validation_df["text"] = validation_df[text_column]
    validation_df["labels"] = validation_df[label_column]
    logger.info(f"Sample training labels: {training_df['labels'].loc[0:3]}")

    if oversample:
        training_df = oversample_reaction_minority_dataset(training_df[["text", "labels"]])
        training_df.to_csv(output_model_path / "oversampled_training_df.csv", index=False)
        random.seed(42)
        logger.info("Oversampled training dataset.")

    logger.info("Getting labels and transforming data... Done.")

    max_len = get_associated_max_len(pretrained_model)
    training_dataset = MultiLabelDataset(training_df, tokenizer, max_len, augment=augment)
    testing_dataset = MultiLabelDataset(validation_df, tokenizer, max_len, augment=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=pretrained_model,
        num_labels=1,  # regression
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )

    if freeze_encoder:
        logger.info("Freezing encoder weights.")
        for param in model.base_model.parameters():
            param.requires_grad = False

    n_params_train = n_params_tot = 0
    for p in model.parameters():
        if p.requires_grad:
            n_params_train += p.numel()
        n_params_tot += p.numel()
    logger.info(f"Number of total parameters in the model: {n_params_tot}.")
    logger.info(f"Number of trainable parameters in the model: {n_params_train}.")

    training_args = TrainingArguments(
        output_dir=str(output_model_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=valid_batch_size,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        overwrite_output_dir=True,
        logging_steps=logging_steps,
        save_total_limit=2,
        save_strategy="epoch",
        load_best_model_at_end=False,
    )

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}."
    )

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Regression metrics
        mse = mean_squared_error(labels, predictions)
        rmse = mean_squared_error(labels, predictions, squared=False)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        smape = (
            1
            / len(labels)
            * np.sum(
                2 * np.abs(predictions - labels) / (np.abs(labels) + np.abs(predictions)) * 100
            )
        )

        predictions = [1 if pred > 0.5 else 0 for pred in predictions]
        logger.info(f"Val labels sample : {labels[0:3]}")
        logger.info(f"Val predictions sample : {predictions[0:3]}")

        # Get weighted accuracy based on frequency
        pos_predictions = [predictions[i] for i in range(len(predictions)) if labels[i] == 1]
        neg_predictions = [predictions[i] for i in range(len(predictions)) if labels[i] == 0]
        pos_labels = [1 for _ in range(len(pos_predictions))]
        neg_labels = [0 for _ in range(len(neg_predictions))]
        pos_acc = np.sum([pred == lab for pred, lab in zip(pos_predictions, pos_labels)]) / len(
            pos_predictions
        )
        neg_acc = np.sum([pred == lab for pred, lab in zip(neg_predictions, neg_labels)]) / len(
            neg_predictions
        )

        balanced_accuracy = balanced_accuracy_score(labels, predictions)
        accuracy = metric.compute(predictions=predictions, references=labels)
        return {
            **accuracy,
            "balanced_accuracy": balanced_accuracy,
            "pos_accuracy": pos_acc,
            "neg_accuracy": neg_acc,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "smape": smape,
        }

    class CustomTrainer(Trainer):
        # Needed to perform a balanced version of the loss
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss (WEIGHTS are computed above)
            batch_weights = (
                torch.where(labels == 0, WEIGHTS[0], WEIGHTS[1])
                .view(-1, self.model.config.num_labels)
                .detach()
            )
            loss_fct = nn.BCEWithLogitsLoss(batch_weights).to(model.device)
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels),
                labels.view(-1, self.model.config.num_labels),
            )
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=testing_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_model_path))
    tokenizer.save_pretrained(output_model_path)

    # Save tokenizer in the checkpoints
    for checkpoint in Path(training_args.output_dir).glob("checkpoint*"):
        tokenizer.save_pretrained(checkpoint)


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--pretrained_model",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the model to fine-tune.",
)
@click.option(
    "--training_data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the training data.",
)
@click.option(
    "--validation_data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the validation data.",
)
@click.option(
    "--rxn_column",
    type=str,
    default="rxn",
    help="Column name of the reaction SMILES column.",
)
@click.option(
    "--rxn_class_column",
    type=str,
    default="rxn_class",
    help="Column name of the reaction class column.",
)
@click.option(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size.",
)
@click.option(
    "--valid_batch_size",
    type=int,
    default=32,
    help="Batch size for the validation set.",
)
@click.option(
    "--epochs",
    type=int,
    default=10,
    help="Number of epochs to train.",
)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-04,
    help="Learning rate.",
)
@click.option(
    "--output_model_path",
    type=click.Path(writable=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the output model file.",
)
@click.option(
    "--weight_decay",
    type=float,
    default=0.001,
    help="Weight decay for the model.",
)
@click.option(
    "--dropout",
    type=float,
    default=0.01,
    help="Dropout for the model.",
)
@click.option(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Path to the checkpoint.",
)
@click.option(
    "--rebalance",
    type=bool,
    is_flag=True,
    help="Wheather to add weights to an imbalanced dataset",
)
@click.option(
    "--smoothing",
    type=float,
    default=None,
    help="Smoothing parameter to perform a gentler rebalancing.",
)
@click.option(
    "--weights",
    type=float,
    nargs=2,
    help="Weights list",
)
@click.option(
    "--freeze_encoder",
    type=bool,
    is_flag=True,
    help="If provided, will freeze the encoder params and train only the last layer.",
)
@click.option(
    "--augment",
    type=bool,
    is_flag=True,
    help="Wheather to perform rotated smiles augmentation on the dataset",
)
@click.option(
    "--logging_steps",
    type=int,
    default=500,
    help="Logging interval.",
)
@click.option(
    "--warmup_steps",
    type=int,
    default=500,
    help="Number of warmup steps for learning rate.",
)
@click.option(
    "--oversample",
    type=bool,
    is_flag=True,
    help="Whether to oversample the minority dataset. Better not to rebalance in this case",
)
def main(
    pretrained_model: Path,
    training_data: Path,
    validation_data: Path,
    rxn_column: str,
    rxn_class_column: str,
    batch_size: int,
    valid_batch_size: int,
    epochs: int,
    learning_rate: float,
    output_model_path: Path,
    weight_decay: float,
    dropout: float,
    resume_from_checkpoint: Optional[str],
    rebalance: bool,
    smoothing: Optional[float],
    weights: List[float],
    freeze_encoder: bool,
    augment: bool,
    logging_steps: int,
    warmup_steps: int,
    oversample: bool,
) -> None:
    """
    Fine-tune a pretrained model on a reaction classification task.
    """
    setup_console_logger()
    logger.info(f'Starting fine-tuning of model "{pretrained_model}".')

    logger.info(f'Loading the tokenizer from "{pretrained_model}"...')
    tokenizer = SmilesTokenizer.from_pretrained(pretrained_model)
    logger.info(f'Loading the tokenizer from "{pretrained_model}"... Done.')

    logger.info(f'Loading the training data from "{training_data}"...')
    training_df = pd.read_csv(training_data)
    logger.info(f'Loading the training data from "{training_data}"... Done.')

    logger.info(f'Loading the validation data from "{validation_data}"...')
    validation_df = pd.read_csv(validation_data)
    logger.info(f'Loading the validation data from "{validation_data}"... Done.')
    if augment:
        logger.info("Training dataset will be augmented during epochs.")

    finetune(
        pretrained_model=pretrained_model,
        tokenizer=tokenizer,
        training_df=training_df,
        validation_df=validation_df,
        text_column=rxn_column,
        label_column=rxn_class_column,
        output_model_path=output_model_path,
        epochs=epochs,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
        resume_from_checkpoint=resume_from_checkpoint,
        rebalance=rebalance,
        smoothing=smoothing,
        weights=weights,
        freeze_encoder=freeze_encoder,
        augment=augment,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        oversample=oversample,
    )


if __name__ == "__main__":
    main()
