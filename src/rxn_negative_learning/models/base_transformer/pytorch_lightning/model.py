"""Pytorch Lightning implementation for VanillaTransformer."""

import logging
from argparse import ArgumentParser
from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# import sentencepiece as _sentencepiece
import torch.optim as optim
from rxn.chemutils.tokenization import detokenize_smiles
from rxn_negative_learning.utils.smiles_utils import lazy_canonicalize_smiles
from torch import Tensor

# sentencepiece has to be loaded before lightning to avoid segfaults
# _sentencepiece
from ...tokenization import SmilesTokenizer
from ..vanilla_transformer.configuration import VanillaTransformerConfig
from ..vanilla_transformer.model import VanillaTransformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class LitVanillaTransformer(pl.LightningModule):
    """Pytorch lightning model for VanillaTransformer."""

    def __init__(
        self,
        model_args: Dict[str, Union[float, int, str]],
        tokenizer: SmilesTokenizer,
    ) -> None:
        """Construct an LM lightning module.

        Args:
            model_args: model's arguments.
        """
        super().__init__()

        self.save_hyperparameters()

        self.model_args = model_args
        logger.info("In __init__")
        logger.info(f"Model args: {self.model_args}")
        logger.info(f"LR: {self.model_args['learning_rate']}")

        self.model: VanillaTransformer
        self.tokenizer = tokenizer

        # Used to log metrics
        self.validation_step_inputs = []
        self.validation_step_outputs = []
        self.validation_step_raw_outputs = []
        self.validation_step_labels = []
        self.validation_step_ids = []

        # Log metrics train
        # Used to log metrics
        self.train_step_inputs = []
        self.train_step_outputs = []
        self.train_step_raw_outputs = []
        self.train_step_labels = []
        self.train_step_scores = []
        self.train_step_rewards = []
        self.train_step_ids = []
        self.train_step_baseline_predictions = []
        self.train_step_baseline_targets = []
        self.init_model()

    def init_model(self) -> None:
        """Initialize a VanillaTransformer."""
        logger.info("In init_model")

        config_args = {
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.sep_token_id,
            "bos_token_id": self.tokenizer.cls_token_id,
            "decoder_start_token_id": self.tokenizer.cls_token_id,
            "vocabulary_size": self.tokenizer.vocab_size,
            "embedding_dim": self.model_args["embedding_dim"],
            "ffnn_hidden_dim": self.model_args["ffnn_hidden_dim"],
            "dropout": self.model_args["dropout"],
            "activation": self.model_args["activation"],
            "num_attention_heads": self.model_args["num_attention_heads"],
            "num_encoder_layers": self.model_args["num_encoder_layers"],
            "num_decoder_layers": self.model_args["num_decoder_layers"],
            "init_std": self.model_args["init_std"],
            "max_position_embeddings": self.model_args["max_position_embeddings"],
            "num_beams": self.model_args["num_beams"],
            "no_teacher_forcing": self.model_args["no_teacher_forcing"],
        }

        if self.model_args["model_name_or_path"] is not None:
            self.model = VanillaTransformer.from_pretrained(
                self.model_args["model_name_or_path"],
            )
            logger.info(f"Model from pretrained: {self.model_args['model_name_or_path']}.")
        else:
            if self.model_args["model_config_name"] is not None:
                config = VanillaTransformerConfig.from_pretrained(
                    self.model_args["model_config_name"]
                )
                logger.info(
                    f"Configuration from pretrained: {self.model_args['model_config_name']}."
                )
            else:
                config = VanillaTransformerConfig(**config_args)
                logger.info("Default configuration.")

            self.model = VanillaTransformer(config)

            logger.info("Training from scratch")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        """Forward pass of the model.

        Raises:
            NotImplementedError: implement this function for the prediction step.
        """
        raise NotImplementedError("Implement the forward function for the LitVanillaTransformer.")

    def configure_optimizers(
        self,
    ) -> Dict[str, object]:
        """Create and return the optimizer.

        Returns:
            output (dict of str: Any):
                - optimizer: the optimizer used to update the parameter.
        """
        if not isinstance(self.model_args["learning_rate"], float):
            raise ValueError("Learning rate should be float")

        # definition of the optimizer
        optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.model_args["learning_rate"],  # type: ignore
            betas=(self.model_args["adam_beta1"], self.model_args["adam_beta2"]),  # type: ignore
            eps=self.model_args["adam_epsilon"],  # type: ignore
            weight_decay=self.model_args["adam_weight_decay"],  # type: ignore
        )

        scheduler = (
            self.model_args["lr_scheduler"]
            if self.model_args["lr_scheduler"] in SCHEDULERS
            else None
        )
        if scheduler is not None:
            logger.info(f"{self.model_args['lr_scheduler']} scheduler is set")

            if scheduler == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

            if scheduler == "linear":
                scheduler = optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=self.model_args["linear_lr_end_factor"],
                    total_iters=self.model_args["linear_lr_total_iters"],
                    verbose=True,
                )  # decreasing linearly until 0.0

            if scheduler == "polynomial":
                scheduler = optim.lr_scheduler.PolynomialLR(
                    optimizer,
                    total_iters=self.model_args["max_steps"],
                    power=self.model_args["polynomial_lr_power"],
                    verbose=True,
                )

            if scheduler == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.model_args["cosine_lr_t_max"]
                )

            if self.model_args["warmup_steps"] > 1:
                warmup = optim.lr_scheduler.LinearLR(
                    optimizer, total_iters=self.model_args["warmup_steps"], verbose=True
                )  # increasing linearly

                scheduler = optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, scheduler],
                    milestones=[self.model_args["warmup_steps"]],
                )
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "interval": "step"}

            # [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 500}]

        logger.info("No lr scheduler found.")
        return {"optimizer": optimizer}  # [optimizer]  # constant lr

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Training step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        loss = self.model(**batch).loss  # type:ignore

        current_learning_rate = self.trainer.optimizers[0].param_groups[0]["lr"]

        self.log("learning_rate", current_learning_rate, prog_bar=False, on_step=True)
        self.log("train_loss", loss)
        return loss

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        print("Outputs type: ", type(outputs))

        # calculating average loss
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        # logging histograms
        self.custom_histogram_adder()
        self.log("avg_loss_train", avg_loss)

    def create_end_of_sequence_mask(self, predictions_ids: torch.Tensor):
        # puts to 0 positions of eos token and everything else to 1
        eos_positions = torch.where(predictions_ids != self.model.config.eos_token_id, 1, 0)
        # replaces all that occurs after the first eos_token (=0) with a 0
        mask = torch.cumprod(eos_positions, dim=1)
        return mask

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Validation step which encompasses the forward pass and the computation of the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            loss computed on the batch.
        """
        output = self.model(**batch)  # type:ignore
        loss = output.loss
        logits = output.logits

        # Get labels
        labels_ids = batch["decoder_input_ids"]
        target_shape = labels_ids.size()
        labels_ids = labels_ids.view(-1, target_shape[-1])

        # Clean predictions
        predictions_ids = torch.argmax(logits, dim=-1)
        end_of_seq_mask = self.create_end_of_sequence_mask(predictions_ids)
        cleaned_predictions_ids = torch.where(
            end_of_seq_mask != 0, predictions_ids, self.model.config.pad_token_id
        )

        predictions = self.tokenizer.batch_decode(
            cleaned_predictions_ids.squeeze(), skip_special_tokens=True
        )
        predictions = [lazy_canonicalize_smiles(detokenize_smiles(pred)) for pred in predictions]
        labels = self.tokenizer.batch_decode(labels_ids.squeeze(), skip_special_tokens=True)
        labels = [detokenize_smiles(lab) for lab in labels]
        self.validation_step_outputs.append(predictions)
        self.validation_step_labels.append(labels)

        accuracy = [1.0 if predictions[i] == labels[i] else 0.0 for i in range(len(predictions))]
        accuracy = np.sum(accuracy) / len(predictions)
        self.log("accuracy_pos_valid", accuracy)
        self.log("loss_valid", loss)
        return loss

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> float:  # type: ignore
        """
        Test step which encompasses the forward pass and the computation of the accuracy and the loss value.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            accuracy computed on the batch.
        """

        input_ids = batch["encoder_input_ids"]
        labels = batch["decoder_input_ids"]

        # generating the predicted sequence
        predictions = self.model.generate(
            input_ids,
            do_sample=False,
            max_length=self.model_args["max_length"],
            num_beams=self.model_args["num_beams"],
        )

        # padding the predicted sequence
        predictions = F.pad(
            predictions,
            pad=(0, self.model_args["max_length"] - predictions.shape[1]),  # type: ignore
            value=self.model.config.pad_token_id,
        )

        # print(self.tokenizer.decode(predictions.squeeze(), skip_special_tokens=True))

        # check if the prediction perfectly matches the labels
        accuracy = 1.0 if torch.equal(predictions.squeeze(), labels.squeeze()) else 0.0

        self.log("accuracy", accuracy, prog_bar=True)

        return accuracy

    def predict_step(self, batch, batch_idx):
        """
        Predict step which encompasses the forward pass and returns the predicted sequences.

        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.

        Returns:
            predicted sequences of the batch.

        link to generate output: https://huggingface.co/docs/transformers/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput.scores
        """
        # TODO: need a way to return the topk predictions

        input_ids = batch["encoder_input_ids"]

        # generating the predicted sequence
        outputs = self.model.generate(
            input_ids,
            do_sample=False,
            max_length=self.model_args["max_length"],
            max_new_tokens=self.model_args["max_length"],
            num_beams=self.model_args["num_beams"],
            output_scores=True,
            return_dict_in_generate=True,
            num_return_sequences=self.model_args["num_predictions_per_sample"],
        )
        logger.info(f"Dict keys: {outputs.keys()}")
        logger.info(f"Model max length: {self.model_args['max_length']}")
        logger.info(f"Sample of sequences: {outputs.sequences[0:3]}")
        logger.info(f"Sample of scores[0]: {outputs.scores[0][0:3]}")
        logger.info(f"Shape of sequences: {outputs.sequences.shape}")
        logger.info(f"Shape of scores[0]: {outputs.scores[0].shape}")
        logger.info(f"Length of scores: {len(outputs.scores)}")

        predictions = outputs.sequences[:, 1:]  # removing class token
        logger.info(f"Sample of sequences: {predictions}")
        scores = torch.stack(outputs.scores, dim=1).softmax(-1)
        logger.info(f"Shape of scores: {scores.shape}")
        logger.info(f"Scores prob sum check: {scores.sum(-1)}")

        # Pad sequences to scores sequence length --> used to manipulate predictions and scores together
        predictions = F.pad(
            predictions,
            pad=(0, scores.shape[1] - predictions.shape[1]),  # type: ignore
            value=self.model.config.pad_token_id,
        )
        logger.info(f"New shape of sequences: {predictions.shape}")

        # # Get rid of scores greater than num_return_sequences. TODO:check if they are ordered
        # log_probs = log_probs[0:predictions.shape[0], :, :]

        # Set scores related to special tokens to 1, as multiplying probs will not account for them
        special_tokens = [
            self.model.config.pad_token_id,
            self.model.config.eos_token_id,
            self.model.config.bos_token_id,
        ]
        logger.info(f"Special tokens ids: {special_tokens}")
        mask = (
            (predictions[:, :, None] != self.model.config.pad_token_id)
            & (predictions[:, :, None] != self.model.config.eos_token_id)
            & (predictions[:, :, None] != self.model.config.bos_token_id)
        )
        probs = torch.where(mask, scores, 1)

        # gather per-token log probabilities
        tok_probs = torch.gather(probs, 2, predictions[:, :, None]).squeeze(-1)
        logger.info(f"Sample of token probs scores: {tok_probs[0:3]}")
        logger.info(f"Computed sequences scores: {tok_probs.prod(-1)}")

        # # normalize across number of sequences: Att! this way the score will change if instead of outputting 3 sequences I
        # # will output 10, because it will normalize across all 3 sequences
        # sum_probs = torch.exp(tok_log_probs).sum(0)
        # logger.info(f"Summed prob tensor: {sum_probs.shape}")
        # tok_log_probs_norm = torch.log(torch.exp(tok_log_probs) / sum_probs)
        # logger.info(f"Shape of log_probs scores normalized: {tok_log_probs_norm.shape}")
        # logger.info(f"Sample of log_probs scores normalized: {tok_log_probs_norm[0:3]}")
        #
        # seq_log_prob_norm = tok_log_probs_norm.sum(-1)
        # logger.info(f"Computed normalized sequences scores: {seq_log_prob_norm}")

        # sequence_scores = outputs.sequences_scores  # seems to be the log prob scores of only the highest probable sequence
        # scores = outputs.scores  # the scores before softmax for each token, for each beam. scores[0] is (n_beams,vocab_size) and collects all the beams for the first token. len(scores) seems to be the maximum length generated within the beams of that sequence and is greater or equal to the predicted sequence length.

        # padding the predicted sequence (do I need this?)
        if len(predictions[0]) == 1:
            predictions = F.pad(
                predictions,
                pad=(0, self.model_args["max_length"] - predictions.shape[1]),  # type: ignore
                value=self.model.config.pad_token_id,
            )
            processed_predictions = self.tokenizer.decode(
                predictions.squeeze(), skip_special_tokens=True
            )
        else:
            processed_predictions = []
            for i in range(len(predictions)):
                processed_predictions.append(
                    self.tokenizer.decode(predictions[i].squeeze(), skip_special_tokens=True)
                )
        logger.info(processed_predictions)
        if self.model_args["num_beams"] == 1:
            sequences_scores = tok_probs.prod(-1)
        else:
            sequences_scores = torch.exp(outputs.sequences_scores)
        return {"predictions": processed_predictions, "scores": sequences_scores}

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Adds model specific arguments to the parser.

        Args:
            parent_parser: argument parser.

        Returns:
            updated parser.
        """

        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # training configuration arguments
        parser.add_argument("--learning_rate", type=float, default=0.0001)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.98)
        parser.add_argument("--adam_epsilon", type=float, default=1e-9)
        parser.add_argument("--adam_weight_decay", type=float, default=0.01)
        parser.add_argument("--model_name_or_path", type=str, default=None)
        parser.add_argument("--model_config_name", type=str, default=None)

        # model configuration arguments
        parser.add_argument("--embedding_dim", type=int, default=256)
        parser.add_argument("--ffnn_hidden_dim", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--activation", type=str, default="relu")
        parser.add_argument("--num_attention_heads", type=int, default=8)
        parser.add_argument("--num_encoder_layers", type=int, default=4)
        parser.add_argument("--num_decoder_layers", type=int, default=4)
        parser.add_argument("--init_std", type=float, default=0.02)
        parser.add_argument("--max_position_embeddings", type=int, default=5000)
        parser.add_argument("--num_beams", type=int, default=1)

        return parser


SCHEDULERS = {"exponential", "linear", "polynomial", "cosine"}
