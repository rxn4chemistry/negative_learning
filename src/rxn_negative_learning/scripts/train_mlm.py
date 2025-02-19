#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Script adapted from Hugging Face library:
https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, CTRL, BERT, RoBERTa, XLNet, Albert).
GPT, GPT-2 and CTRL are fine-tuned using a causal language modeling (CLM) loss. BERT, RoBERTa and Albert are fine-tuned
using a masked language modeling (MLM) loss. XLNet is fine-tuned using a permutation language modeling (PLM) loss.
Albert is employed for the RXNMapper or rxnfp training.
"""

import logging
import math
import os
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Optional, Type

import transformers
from torch.utils.data import ConcatDataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithRefDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process

from rxn_negative_learning.models.tokenization import SmilesTokenizer

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from"
                " scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    vocab_path: Optional[str] = field(default=None, metadata={"help": "Vocabulary path"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The input training data files (multiple files in glob format). "
                "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
            )
        },
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    train_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={
            "help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."
        },
    )

    mlm: bool = field(
        default=False,
        metadata={"help": "Train with masked-language modeling loss instead of language modeling."},
    )
    whole_word_mask: bool = field(
        default=False, metadata={"help": "Whether ot not to use whole word mask."}
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss"},
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": (
                "Ratio of length of a span of masked tokens to surrounding context length for permutation language"
                " modeling."
            )
        },
    )
    max_span_length: int = field(
        default=5,
        metadata={
            "help": "Maximum length of a span of masked tokens for permutation language modeling."
        },
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization."
                "The training dataset will be truncated in block of this size for training."
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


def get_dataset(
    args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
):
    def _dataset(file_path, ref_path=None):
        if args.line_by_line:
            if ref_path is not None:
                if not args.whole_word_mask or not args.mlm:
                    raise ValueError(
                        "You need to set world whole masking and mlm to True for Chinese Whole Word Mask"
                    )
                return LineByLineWithRefDataset(
                    tokenizer=tokenizer,
                    file_path=file_path,
                    block_size=args.block_size,
                    ref_path=ref_path,
                )

            return LineByLineTextDataset(
                tokenizer=tokenizer, file_path=file_path, block_size=args.block_size
            )
        else:
            return TextDataset(
                tokenizer=tokenizer,
                file_path=file_path,
                block_size=args.block_size,
                overwrite_cache=args.overwrite_cache,
                cache_dir=cache_dir,
            )

    if evaluate:
        return _dataset(args.eval_data_file, args.eval_ref_file)
    elif args.train_data_files:
        return ConcatDataset([_dataset(f) for f in glob(args.train_data_files)])
    else:
        return _dataset(args.train_data_file, args.train_ref_file)


def main():
    # See all possible arguments i
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    train_mlm(model_args, data_args, training_args)


def train_mlm(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    tokenizer_cls: Optional[Type[PreTrainedTokenizer]] = None,
) -> None:
    """
    Common code for pre-training language-based transformer models.
    Args:
        model_args: model options.
        data_args: dataset options.
        training_args: trainer options.
        tokenizer_cls: tokenizer class - must be instantiatable from the vocab path.
    """
    if tokenizer_cls is None:
        tokenizer_cls = SmilesTokenizer

    if model_args.vocab_path is None:
        raise ValueError('"--vocab_path" must be specified!')
    tokenizer = tokenizer_cls(model_args.vocab_path)

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use"
            " --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}."
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if (
        config.model_type in ["bert", "roberta", "distilbert", "camembert", "albert"]
        and not data_args.mlm
    ):
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

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

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        get_dataset(
            data_args,
            tokenizer=tokenizer,
            evaluate=True,
            cache_dir=model_args.cache_dir,
        )
        if training_args.do_eval
        else None
    )
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
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

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # prediction_loss_only=True,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None
            and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(resume_from_checkpoint=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    # Save tokenizer in the checkpoints
    for checkpoint in Path(training_args.output_dir).glob("checkpoint*"):
        tokenizer.save_pretrained(checkpoint)

    # TODO: what to do with this?
    print(results)


if __name__ == "__main__":
    main()

# scorer-train-mlm --config_name /Users/ato/Desktop/Git/rxn_negative_learning/data/scorer_config.json --train_data_file /Users/ato/Desktop/Git/rxn_negative_learning/data/usptoold/0.1split-ratio/data-train.txt --eval_data_file /Users/ato/Desktop/Git/rxn_negative_learning/data/usptoold/0.1split-ratio/data-valid.txt --vocab_path /Users/ato/Desktop/Git/rxn_negative_learning/data/usptoold/joint-vocab-usptoold-regiosqm.txt --output_dir /tmp/ --evaluation_strategy "steps" --eval_steps 5000 --save_steps 5000 --line_by_line --mlm --do_train --do_eval --learning_rate 0.0001 --logging_steps 2500 --per_device_eval_batch_size 16 --per_device_train_batch_size 16 --save_total_limit 200 --num_train_epochs 100
