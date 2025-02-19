# Code from rxn_transformers: https://github.ibm.com/rxn/rxn-transformers

import logging
from pathlib import Path
from typing import Type

import click
from datasets import load_metric
from rxn.utilities.logging import setup_console_logger
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.scripts.train_mlm import DataTrainingArguments, get_dataset

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def eval_mlm(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    batch_size: int,
    tokenizer_cls: Type[PreTrainedTokenizer],
) -> None:
    """
    Evaluate a language model.
    Most of the code is similar to the one for launching a training, as the
    easiest way to evaluate a model, it seems, is to replicate the training
    conditions.
    Args:
        model_path: path to the pretrained model
        data_path: path to the TXT on which to evaluate the MLM accuracy.
        output_dir: where to save the evaluation results.
        batch_size: batch size.
        tokenizer_cls: tokenizer class - must be instantiatable from the vocab path.
    """
    data_args = DataTrainingArguments(
        eval_data_file=str(data_path),
        line_by_line=True,
        mlm=True,
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=batch_size,
    )

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}."
    )

    # Load the model
    tokenizer = tokenizer_cls.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)

    if data_args.block_size <= 0:
        # Note: tokenizer.model_max_length may be 1000000000000000019884624838656,
        # so we also look at what comes from the model config.
        data_args.block_size = min(tokenizer.model_max_length, model.config.max_position_embeddings)
    else:
        data_args.block_size = min(
            data_args.block_size,
            tokenizer.model_max_length,
            model.config.max_position_embeddings,
        )

    # Load the data
    eval_dataset = get_dataset(
        data_args,
        tokenizer=tokenizer,
        evaluate=True,
        cache_dir=None,
    )
    if data_args.mlm and data_args.whole_word_mask:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer=tokenizer, mlm_probability=data_args.mlm_probability
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=data_args.mlm,
            mlm_probability=data_args.mlm_probability,
        )

    # Metrics

    metric = load_metric("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    def preprocess_logits_for_metrics(logits, _):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    predictions = trainer.predict(test_dataset=eval_dataset)
    print(predictions)


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the model to evaluate",
)
@click.option(
    "--eval_data_file",
    "-d",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to evaluation data TXT",
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    required=True,
    help="Path to the output directory",
)
@click.option(
    "--batch_size",
    "-b",
    type=int,
    default=32,
    help="Batch size for evaluation",
)
def main(
    model: Path,
    eval_data_file: Path,
    output_dir: Path,
    batch_size: int,
) -> None:
    setup_console_logger()

    eval_mlm(
        model_path=model,
        data_path=eval_data_file,
        output_dir=output_dir,
        batch_size=batch_size,
        tokenizer_cls=SmilesTokenizer,
    )


if __name__ == "__main__":
    main()
