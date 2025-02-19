import copy
import json
import logging
from itertools import takewhile
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from rxn.chemutils.tokenization import detokenize_smiles
from rxn_negative_learning.data_generation.data_standardizer import SMILESDataStandardizer
from rxn_negative_learning.models.base_transformer.pytorch_lightning.model import (
    LitVanillaTransformer,
)
from rxn_negative_learning.models.base_transformer.vanilla_transformer.configuration import (
    VanillaTransformerConfig,
)
from rxn_negative_learning.models.base_transformer.vanilla_transformer.model import (
    VanillaTransformer,
)
from rxn_negative_learning.models.baseline.baseline_model import BASELINE_MODELS, BaselineTrainer
from rxn_negative_learning.models.scorers.scorer_base import RXNNegScorerBase
from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.utils.smiles_utils import partial_smiles_sequences
from rxn_negative_learning.utils.targets_dict import NEG_TARGETS_DICT
from torch import Tensor, nn
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax


def dump_list_to_file(values, filename) -> None:
    """Write an iterable of strings to a file.

    Args:
        values: values to write to the file.
        filename: file to write to. Will be overwritten if it exists already.
    """
    with open(filename, "wt") as f:
        for v in values:
            f.write(f"{v}\n")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ReinforceLitVanillaTransformer(LitVanillaTransformer):
    def __init__(
        self,
        model_args: Dict[str, Union[float, int, str, Any]],
        tokenizer: SmilesTokenizer,
        scorer: RXNNegScorerBase,
    ) -> None:
        """Construct an LM lightning module with RL training capabilities.
        Args:
            model_args: model's arguments.
        """
        super(ReinforceLitVanillaTransformer, self).__init__(model_args, tokenizer)
        self.save_hyperparameters()
        self.nll_loss = nn.NLLLoss(ignore_index=self.model.config.pad_token_id, reduction="none")

        # TODO: Initialize baseline and reference model from checkpoint
        # Initialize baseline model
        self.baseline_model = None
        if BASELINE_MODELS.get(self.model_args["with_baseline"], None) is not None:
            logger.info(f"Initializing baseline model: {self.model_args['with_baseline']}")
            b_model = BASELINE_MODELS[self.model_args["with_baseline"]](
                self.model_args["embedding_dim"], self.model_args["baseline_model_dropout"]
            )
            self.baseline_model = BaselineTrainer(
                b_model,
                learning_rate=self.model_args["baseline_model_lr"],
                weight_decay=self.model_args["baseline_model_weight_decay"],
                oversampling_threshold=self.model_args["baseline_model_oversampling_threshold"],
                device=self.model.device,
            )

        self.BASELINE_TARGETS_DICT = None
        if self.model_args["baseline_targets_file"] is not None:
            logger.info(
                f"Using lookup table for baseline: {self.model_args['baseline_targets_file']}"
            )
            with open(self.model_args["baseline_targets_file"]) as f:
                self.BASELINE_TARGETS_DICT = json.load(f)

        # Initialize scorer / reward model
        self.scorer = scorer

        # Initialize reference model
        self.reference_model = None
        self.regularization_beta = self.model_args[
            "regularization_beta"
        ]  # weights the regularization term
        self.kl_loss = KLDivLoss(log_target=True, reduction="none")
        if self.model_args["regularization"]:
            logger.info(f"Initializing regularization model: {self.model_args['regularization']}")
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

            # Initialize reference model
            base_config = VanillaTransformerConfig(**config_args)
            self.reference_model = VanillaTransformer(base_config).to(self.device)
            self.reference_model.load_state_dict(copy.deepcopy(self.model.state_dict()))

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Training step which encompasses the forward pass,
        the evaluation of the reward on the` predicted sequences
        and the computation of the loss value.
        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.
        Returns:
            loss computed on the batch.
        """
        # Save for statistics and metrics
        self.train_step_scores.append(batch["score"])
        self.train_step_ids.append(batch["idx"])

        # Get logits from the batch
        logits = self.model(**batch).logits

        # Compute log probabilities
        log_probs = log_softmax(logits, dim=-1)
        logger.info(f"Log probabilities shape: {log_probs.shape}")

        # Get the predicted ids and create end-of-sequence mask excluding everything after first EOS token
        # and putting it to PAD token id
        predictions_ids_raw = torch.argmax(log_probs, dim=-1).detach()
        end_of_seq_mask = self.create_end_of_sequence_mask(predictions_ids_raw).detach()
        logger.info(f"EOS mask shape: {end_of_seq_mask.shape}")
        predictions_ids = torch.where(
            end_of_seq_mask != 0, predictions_ids_raw, self.model.config.pad_token_id
        )

        # Extract the predictions' tokens and standardize the predictions -> returns "" for invalid smiles
        raw_predictions = self.tokenizer.batch_decode(
            predictions_ids.squeeze(), skip_special_tokens=True
        )
        logger.info(f"Sample of raw predictions without special tokens:{raw_predictions[0:3]}")
        predictions = [
            SMILESDataStandardizer().standardize(detokenize_smiles(pred))
            for pred in raw_predictions
        ]
        logger.info(f"Sample of canonical predictions without special tokens:{predictions[0:3]}")

        # Get the input/conditioning to construct the example for the reward model
        # Standardization is needed because the input can be augmented
        conditioning = batch["encoder_input_ids"]
        conditioning = self.tokenizer.batch_decode(conditioning.squeeze(), skip_special_tokens=True)
        conditioning = [
            SMILESDataStandardizer().standardize(detokenize_smiles(cond)) for cond in conditioning
        ]

        # Combine canonical predictions and conditioning and compute the list of SEQUENCE reward
        conditioned_predictions = self.combine_conditioning_with_predictions(
            conditioning, predictions
        )  # returns 'ABC>>' for invalids
        rewards = self.get_reward(conditioned_predictions)
        self.train_step_rewards.append(rewards)
        logger.info(f"Sample of canonical conditioned predictions:{conditioned_predictions[0:3]}")
        logger.info(f"Sample of sequence rewards: {rewards[0:3]}")

        self.log("reward/avg_train", np.mean(np.array(rewards)).item(), on_epoch=True)
        self.log("reward/std_train", np.std(np.array(rewards)).item(), on_epoch=True)

        # Create a tensor of advantages, one for each token: initially is the SEQUENCE reward expanded to each token
        reward_tensor = torch.unsqueeze(torch.FloatTensor(rewards), 1)  # (batch_size X 1)
        batch_size = reward_tensor.shape[0]
        reward_tensor = reward_tensor.expand(
            -1, self.model.decoder_outputs.shape[1]
        )  # (batch_size X max_sequence_len)
        reward_tensor = torch.reshape(
            reward_tensor, (reward_tensor.shape[-1] * reward_tensor.shape[0],)
        ).to(self.model.device)  # 1D tensor (batch_size*max_seq_len X 1)
        end_of_seq_mask = torch.reshape(
            end_of_seq_mask, (end_of_seq_mask.shape[-1] * end_of_seq_mask.shape[0],)
        ).to(self.model.device)  # 1D tensor (batch_size*max_seq_len X 1)

        # If a baseline is provided modify the advantage tensor accordingly and train baseline
        if self.baseline_model is not None:
            logger.info(f"Applying baseline: {self.model_args['with_baseline']}")

            # 1st Prepare hidden embeddings for baseline model
            input_baseline = torch.roll(
                self.model.decoder_outputs, -1, 1
            )  # moves the first embedding (unused) to the end
            input_baseline = (
                input_baseline.view(1, input_baseline.shape[1] * input_baseline.shape[0], -1)
                .squeeze()
                .to(self.model.device)
            )
            logger.info(
                f"Embeddings size before and after preprocessing for baseline: "
                f"B {self.model.decoder_outputs.shape}, A {input_baseline.shape}"
            )

            # 2nd - Predict on the batch
            baseline_predictions = (
                self.baseline_model.predict(input_baseline, randomic=self.model_args["randomic"])
                .squeeze()
                .to(self.model.device)
            )
            logger.info(f"Sample of baseline predictions: {baseline_predictions[0:3]}")
            logger.info(f"Shape of baseline predictions: {baseline_predictions.shape}")
            self.log_baseline_metrics(baseline_predictions, reward_tensor, end_of_seq_mask)

            logger.info(f"Sample of reward: {reward_tensor[0:3]}")
            advantage = torch.sub(reward_tensor, baseline_predictions)
            logger.info(f"Sample of advantage: {advantage[0:3]}")

            # 2nd - Train baseline
            # input_baseline = self.baseline_model.preprocess_step(input_baseline) CORRECT?
            target_baseline, additional_mask = self.process_baseline_target(
                reward_tensor, raw_predictions, conditioning
            )  # define baseline target
            logger.info(f"Target baseline: {target_baseline}")
            baseline_mask = (
                end_of_seq_mask  # no need to shift because embeddings were shifted already
            )
            if additional_mask is not None:  # excludes partial sequences not found in lookup table
                baseline_mask = end_of_seq_mask * additional_mask

            self.baseline_model.train_step(
                x=input_baseline,
                y=target_baseline,
                mask=baseline_mask,
                batch_size=self.model_args["baseline_batch_size"],
            )

            if (
                len(self.baseline_model.loss_history) > 0
                and len(self.baseline_model.validation_loss_history) > 0
            ):
                self.log("baseline/loss", self.baseline_model.loss_history[-1])
                self.log("baseline/loss_valid", self.baseline_model.validation_loss_history[-1])

            # Save baseline predictions
            self.train_step_baseline_predictions.append(
                baseline_predictions.view(batch_size, -1).tolist()
            )
            self.train_step_baseline_targets.append(target_baseline.view(batch_size, -1).tolist())
        else:
            logger.info("No baseline applied! Advantage is just the expanded reward tensor")
            advantage = reward_tensor

        # Whiten the advantages
        advantage = (
            (advantage - torch.mean(advantage)) / torch.std(advantage)
            if torch.std(advantage) != 0.0
            else advantage
        )
        logger.info(f"Sample of whitened advantage: {advantage[0:3]}")

        # Compute RL loss
        nll_log_probs = self.nll_loss(
            log_probs.reshape(-1, logits.shape[-1]), predictions_ids.reshape(-1)
        )
        rl_loss = torch.mean(
            torch.sum(nll_log_probs * advantage.reshape(-1) * end_of_seq_mask, dim=-1)
        )

        entropy = (
            torch.sum(nll_log_probs * (-nll_log_probs).exp() * end_of_seq_mask)
            / torch.count_nonzero(end_of_seq_mask).detach()
        )
        logger.info(f"Policy entropy: {entropy}")
        logger.info(f"RL loss: {rl_loss}")
        self.log("rl/entropy", entropy)
        self.log("rl/loss_rl_train", rl_loss)
        self.log(
            "rl/loss_mle_train", torch.mean(torch.sum(nll_log_probs * end_of_seq_mask, dim=-1))
        )
        self.log(
            "rl/loss_advantage_train",
            torch.mean(torch.sum(advantage.reshape(-1) * end_of_seq_mask, dim=-1)),
        )

        # Add regularization if specified
        if self.model_args["regularization"] is not None:
            if self.model_args.get("finetune", None) is not None and self.global_step == 0:
                # For finetuning I want the refence model to be the starting model
                # This is not loaded if the base model does not have it (e.g. was trained with MLE)
                self.reference_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            # Update base model every x steps if specified
            if (
                self.model_args["update_regularization_model_every_step"] is not None
                and self.global_step != 0
                and self.global_step % self.model_args["update_regularization_model_every_step"]
                == 0
            ):
                self.reference_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
                logger.info(f"Updated regularization model at step {self.global_step}.")

            self.reference_model.eval()
            reference_batch = {k: v.detach() for k, v in batch.items()}
            reference_logits = self.reference_model(**reference_batch).logits
            reference_log_probs = log_softmax(reference_logits, dim=-1).detach()
            reference_ids = torch.argmax(reference_logits, dim=-1).detach()
            reference_end_of_seq_mask = self.create_end_of_sequence_mask(reference_ids).detach()
            reference_end_of_seq_mask = torch.reshape(
                reference_end_of_seq_mask,
                (reference_end_of_seq_mask.shape[-1] * reference_end_of_seq_mask.shape[0],),
            )

            regularization_loss = self.compute_regularization_loss(
                reference_log_probs, log_probs, predictions_ids_raw, end_of_seq_mask
            )
            rl_loss += regularization_loss
            logger.info(f"RL loss + regularization: {rl_loss}")
        else:
            logger.info("No regularization applied!")
        # # Save train data
        # Get targets
        labels_ids = batch["decoder_input_ids"]
        target_shape = labels_ids.size()
        labels_ids = labels_ids.view(-1, target_shape[-1])
        labels = self.tokenizer.batch_decode(labels_ids.squeeze(), skip_special_tokens=True)
        labels = [detokenize_smiles(lab) for lab in labels]

        # Store
        self.train_step_raw_outputs.append(raw_predictions)
        self.train_step_outputs.append(predictions)
        self.train_step_inputs.append(conditioning)
        self.train_step_labels.append(labels)

        return rl_loss

    def get_reward(self, predictions: List[str]) -> List[float]:
        rewards = self.scorer(predictions)
        return rewards

    def combine_conditioning_with_predictions(
        self, conditioning: List[str], predictions: List[str]
    ):
        return [f"{conditioning[i]}>>{predictions[i]}" for i in range(len(predictions))]

    def stop_at_end_of_sequence(self, predictions: List[str]):
        """
        Needed to stop the predictions when the first end-of-sequence token is hit
        """
        new_predictions = []
        for pred in predictions:
            new_pred = list(
                takewhile(
                    lambda x: x
                    != self.tokenizer.convert_ids_to_tokens(self.model.config.eos_token_id),
                    pred.split(" "),
                )
            )
            new_predictions.append(" ".join(new_pred))
        return new_predictions

    def create_end_of_sequence_mask(self, predictions_ids: torch.Tensor):
        # puts to 0 positions of eos token and everything else to 1
        eos_positions = torch.where(predictions_ids != self.model.config.eos_token_id, 1, 0)
        # replaces all that occurs after the first eos_token (=0) with a 0
        mask = torch.cumprod(eos_positions, dim=1)
        return mask

    def update_lookup_table_with_rewards(self, reward_list, partial_predictions):
        logger.info(f"Updating lookup table: epoch {self.current_epoch} | step {self.global_step}")
        for i, elem in enumerate(partial_predictions):
            for p in elem:
                if p in self.BASELINE_TARGETS_DICT.keys():
                    self.BASELINE_TARGETS_DICT[p]["score"] += reward_list[i]
                    self.BASELINE_TARGETS_DICT[p]["count"] += 1
                else:
                    self.BASELINE_TARGETS_DICT[p] = {"score": reward_list[i], "count": 1}
        logger.info("Updating lookup table ... Done")

    def process_baseline_target(self, reward_tensor, raw_predictions, conditioning):
        # Get the lookup table for the baseline if provided
        if self.model_args["baseline_targets_file"] is not None:
            # For now it puts the baseline target to -100 (not counted) if the partial sequence is not found
            partial_predictions = [
                [f"{cond}>>{p}" for p in partial_smiles_sequences(pred)]
                for pred, cond in zip(raw_predictions, conditioning)
            ]
            logger.info(f"Sample of raw partial predictions:{partial_predictions[0][0:10]}")
            partial_predictions_scores = [
                torch.tensor([
                    self.BASELINE_TARGETS_DICT[p]["score"]
                    / float(self.BASELINE_TARGETS_DICT[p]["count"])
                    if p in self.BASELINE_TARGETS_DICT.keys()
                    else -100
                    for p in pred
                ])
                for pred in partial_predictions
            ]
            logger.info(
                f"Sample of partial predictions scores:{partial_predictions_scores[0][0:10]}"
            )

            partial_predictions_frequency = [
                [1 if p in self.BASELINE_TARGETS_DICT.keys() else 0 for p in pred]
                for pred in partial_predictions
            ]  # used for logging

            avg_lookup_frequency = np.mean([np.mean(p) for p in partial_predictions_frequency])
            self.log("baseline/avg_lookup_frequency", avg_lookup_frequency.item(), on_epoch=True)

            # Pad the baseline value for the sequences to -100
            max_len = self.model.decoder_outputs.shape[1]
            partial_predictions_scores = [
                torch.nn.functional.pad(
                    x, pad=(0, max_len - x.numel()), mode="constant", value=-100
                )
                for x in partial_predictions_scores
            ]
            partial_predictions_scores = torch.stack(partial_predictions_scores)
            partial_predictions_scores_mask = torch.where(partial_predictions_scores > -100, 1, 0)

            baseline_target = torch.reshape(
                partial_predictions_scores,
                (partial_predictions_scores.shape[-1] * partial_predictions_scores.shape[0],),
            ).to(self.model.device)
            logger.info(f"Baseline Target: {baseline_target}")
            return baseline_target, partial_predictions_scores_mask.view(-1, 1).squeeze().to(
                self.model.device
            )
        return reward_tensor, None

    def log_baseline_metrics(self, baseline_predictions, reward_tensor, mask):
        # Log overall values for baseline: mean and std for all sequences (exclude with mask everything after EOS)
        baseline_mean = torch.sum(baseline_predictions * mask).item() / torch.count_nonzero(mask)
        baseline_std = torch.sqrt(
            torch.sum(torch.pow((baseline_predictions - baseline_mean) * mask, 2)).item()
            / torch.count_nonzero(mask)
        )

        self.log("baseline/avg_predicted_value", baseline_mean)
        self.log("baseline/std_predicted_value", baseline_std)

        # Log values for baseline by positives and negatives: mean and std for all sequences (exclude after EOS)
        # Positives
        scores = torch.where(reward_tensor >= 0.5, 1, 0)
        n_positives = torch.count_nonzero(mask * scores)
        pos_baseline_predictions = baseline_predictions * mask * scores
        pos_baseline_mean = torch.sum(pos_baseline_predictions).item() / n_positives
        pos_baseline_std = torch.sqrt(
            torch.sum(
                torch.pow((pos_baseline_predictions - pos_baseline_mean) * mask * scores, 2)
            ).item()
            / n_positives
        )

        self.log("baseline/avg_predicted_value_pos", pos_baseline_mean)
        self.log("baseline/std_predicted_value_pos", pos_baseline_std)

        # Negatives
        inverted_scores = torch.where(reward_tensor < 0.5, 1, 0)
        n_negatives = torch.count_nonzero(mask * inverted_scores)  # this includes also invalids
        neg_baseline_predictions = baseline_predictions * mask * inverted_scores
        neg_baseline_mean = torch.sum(neg_baseline_predictions).item() / n_negatives
        neg_baseline_std = torch.sqrt(
            torch.sum(
                torch.pow(
                    (neg_baseline_predictions - neg_baseline_mean) * mask * inverted_scores, 2
                )
            ).item()
            / n_negatives
        )

        self.log("baseline/avg_predicted_value_neg", neg_baseline_mean)
        self.log("baseline/std_predicted_value_neg", neg_baseline_std)

    def compute_regularization_loss(
        self, base_log_probs, log_probs, predictions_ids, end_of_sequence_mask
    ):
        def reg_loss(policy, ref_policy, regularization_type):
            logger.info(f"log probs shape: P {policy.shape} , R {ref_policy.shape}")
            if regularization_type == "kl-abs":
                return torch.abs(self.kl_loss(policy, ref_policy))  # forward
            elif regularization_type == "jsd":
                return (self.kl_loss(policy, ref_policy) + self.kl_loss(ref_policy, policy)) / 2
            elif regularization_type == "kl-forward":
                return self.kl_loss(policy, ref_policy)
            elif regularization_type == "kl-reverse":
                return self.kl_loss(ref_policy, policy)
            else:
                raise ValueError(f"'{regularization_type}' is not a valid regularization argument!")

        loss = reg_loss(log_probs, base_log_probs, self.model_args["regularization"])
        loss = loss.reshape(-1, log_probs.shape[-1])
        logger.info(f"Reg loss shape: {loss.shape}")

        # Masking all the tokens beyond EOS
        masking = torch.tensor(end_of_sequence_mask.unsqueeze(1).expand(-1, log_probs.shape[-1]))
        logger.info(f"Regularization mask shape: {masking.shape}")
        # # Below removing also all the probabilities not corresponding to the predicted token
        # logger.info(f"Pred ids shape: {predictions_ids.shape}")
        # logger.info(f"Pred ids: {predictions_ids}")
        # masking[np.arange(masking.shape[0]), predictions_ids.view(-1).tolist()] = -100
        # masking = torch.where(masking != -100, 0, 1)
        non_zero_elements = torch.count_nonzero(
            end_of_sequence_mask
        ).item()  # average kl for a token
        logger.info(f"Regularization mask number of non-zero elements: {non_zero_elements}")
        logger.info(
            f"Regularization mask number of non-zero elements before: {torch.count_nonzero(masking).item()}"
        )
        # loss = (torch.sum(loss * masking) / log_probs.shape[0])  # divided by the batch size!
        loss = torch.sum(loss * masking) / non_zero_elements

        # Clip beta and make it dynamic
        dynamic_beta = self.model_args.get("dynamic_beta", None)
        if dynamic_beta is not None and dynamic_beta == "clipping":
            regularization_ratio = (
                loss.detach() - self.model_args["loss_regularization_reference"]
            ) / self.model_args["loss_regularization_reference"]
            epsilon = torch.clamp(regularization_ratio, -0.2, 0.2)
            self.regularization_beta = self.regularization_beta * (
                1 + self.model_args["K_beta"] * epsilon
            )

        if dynamic_beta is not None and dynamic_beta == "ppo":
            if loss < self.model_args["loss_regularization_reference"] / 1.5:
                self.regularization_beta = self.regularization_beta / 2
            elif loss > self.model_args["loss_regularization_reference"] * 1.5:
                self.regularization_beta = self.regularization_beta * 2

        self.log(f"regularization/{self.model_args['regularization']}_loss", loss)
        self.log("regularization/beta", self.regularization_beta)
        logger.info(f"{self.model_args['regularization']} regularization loss: {loss}")
        logger.info(f"regularization beta: {self.regularization_beta}")

        return self.regularization_beta * loss

    def on_train_epoch_end(self) -> None:
        """
        After each train epoch log some metrics
        """
        predictions = flatten(self.train_step_outputs)
        raw_outputs = flatten(self.train_step_raw_outputs)
        inputs = flatten(self.train_step_inputs)
        labels = flatten(self.train_step_labels)
        b_predictions = flatten(self.train_step_baseline_predictions)
        b_targets = flatten(self.train_step_baseline_targets)
        scores = flatten(self.train_step_scores)
        rewards = flatten(self.train_step_rewards)
        ids = flatten(self.train_step_ids)

        self.train_step_outputs.clear()
        self.train_step_raw_outputs.clear()
        self.train_step_inputs.clear()
        self.train_step_labels.clear()
        self.train_step_baseline_predictions.clear()
        self.train_step_baseline_targets.clear()
        self.train_step_scores.clear()
        self.train_step_ids.clear()

        # Also update lookup table
        if (
            self.BASELINE_TARGETS_DICT is not None
            and self.model_args["update_lookup_table_every_n_epochs"] is not None
            and self.current_epoch % self.model_args["update_lookup_table_every_n_epochs"] == 0
        ):
            partial_predictions = [
                [f"{cond}>>{p}" for p in partial_smiles_sequences(pred)]
                for pred, cond in zip(raw_outputs, inputs)
            ]
            self.update_lookup_table_with_rewards(rewards, partial_predictions)

        predictions_pos = [predictions[i] for i in range(len(predictions)) if scores[i] == 1]
        labels_pos = [labels[i] for i in range(len(labels)) if scores[i] == 1]
        accuracy_pos = [
            1.0 if predictions_pos[i] == labels_pos[i] else 0.0 for i in range(len(predictions_pos))
        ]
        accuracy_pos = np.sum(accuracy_pos) / len(predictions_pos)
        self.log("accuracy/pos_train", accuracy_pos)

        accuracy_neg = [
            1.0 if predictions[i] in NEG_TARGETS_DICT[ids[i].item()] else 0.0
            for i in range(len(predictions))
        ]
        accuracy_neg = np.sum(accuracy_neg) / len(predictions)
        self.log("accuracy/neg_train", accuracy_neg)

        invalid_ratio = [1.0 if predictions[i] == "" else 0.0 for i in range(len(predictions))]
        self.log("invalid_ratio_train", np.sum(invalid_ratio) / len(predictions))

        # # Saving lookup table
        # output_file_lookup_table = Path(self.model_args["output_dir"]) / "lookup_table"
        # output_file_lookup_table.mkdir(parents=True, exist_ok=True)
        # output_file_lookup_table = output_file_lookup_table / f"lookup-table-step{self.global_step}.json"
        # with open(output_file_lookup_table, 'w') as f:
        #     json.dump(self.BASELINE_TARGETS_DICT, f, indent=4)

        if self.model_args["save_train_predictions"]:
            logger.info(f"Global step: {self.global_step}")
            logger.info("Saving train predictions to file.")
            logger.info(
                "Warning: this code does not consider the model version. Existing files will be overwritten."
            )
            output_dir = Path(self.model_args["output_dir"]) / "train_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            dump_list_to_file(predictions, output_dir / f"predictions-step{self.global_step}.txt")
            dump_list_to_file(
                raw_outputs, output_dir / f"raw-predictions-step{self.global_step}.txt"
            )
            dump_list_to_file(inputs, output_dir / f"inputs-step{self.global_step}.txt")
            dump_list_to_file(labels, output_dir / f"targets-step{self.global_step}.txt")
            dump_list_to_file(
                b_predictions, output_dir / f"b-predictions-step{self.global_step}.txt"
            )
            dump_list_to_file(b_targets, output_dir / f"b-targets-step{self.global_step}.txt")

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:  # type: ignore
        """
        Validation step which encompasses the forward pass and the computation of the loss value.
        Args:
            batch: dictionary containing the input_ids and the attention_type.
            batch_idx: index of the current batch, unused.
        Returns:
            loss computed on the batch.
        """

        # TODO: add negative accuracy

        output = self.model(**batch)  # type:ignore
        loss = output.loss
        logits = output.logits

        # Get conditioning
        conditioning = batch["encoder_input_ids"]
        conditioning = self.tokenizer.batch_decode(conditioning.squeeze(), skip_special_tokens=True)
        conditioning = [detokenize_smiles(cond) for cond in conditioning]
        self.validation_step_inputs.append(conditioning)

        # Get labels
        labels_ids = batch["decoder_input_ids"]
        # logger.info(f"Val label ids: {labels_ids[0:3]}")
        target_shape = labels_ids.size()
        labels_ids = labels_ids.view(-1, target_shape[-1])

        # Clean predictions
        predictions_ids = torch.argmax(logits, dim=-1)
        # logger.info(f"Val prediction ids: {predictions_ids[0:3]}")
        end_of_seq_mask = self.create_end_of_sequence_mask(predictions_ids)
        cleaned_predictions_ids = torch.where(
            end_of_seq_mask != 0, predictions_ids, self.model.config.pad_token_id
        )

        predictions = self.tokenizer.batch_decode(
            cleaned_predictions_ids.squeeze(), skip_special_tokens=True
        )
        self.validation_step_raw_outputs.append(predictions)
        predictions = [
            SMILESDataStandardizer().standardize(detokenize_smiles(pred)) for pred in predictions
        ]
        labels = self.tokenizer.batch_decode(labels_ids.squeeze(), skip_special_tokens=True)
        labels = [detokenize_smiles(lab) for lab in labels]
        logger.info(f"Val predictions: {predictions[0:3]}")
        logger.info(f"Val labels: {labels[0:3]}")
        self.validation_step_outputs.append(predictions)
        self.validation_step_labels.append(labels)

        accuracy = [1.0 if predictions[i] == labels[i] else 0.0 for i in range(len(predictions))]
        accuracy = np.sum(accuracy) / len(predictions)

        self.log("accuracy_pos_valid", accuracy, on_epoch=True)

        ids = batch["idx"]
        self.validation_step_ids.append(ids)

        accuracy_neg = [
            1.0 if predictions[i] in NEG_TARGETS_DICT[ids[i].item()] else 0.0
            for i in range(len(predictions))
        ]
        accuracy_neg = np.sum(accuracy_neg) / len(predictions)
        self.log("accuracy/neg_valid", accuracy_neg, on_epoch=True)

        invalid_ratio = [1.0 if predictions[i] == "" else 0.0 for i in range(len(predictions))]
        self.log("invalid_ratio_valid", np.sum(invalid_ratio) / len(predictions))
        self.log("loss/valid", loss)
        return loss

    def on_validation_epoch_end(self):
        """
        After each validation step log some metrics.
        """
        # num_batches = len(self.trainer.train_dataloader()) / self.trainer.accumulate_grad_batches
        # logger.info(f"Number of training batches: {num_batches}")
        predictions = flatten(self.validation_step_outputs)
        raw_outputs = flatten(self.validation_step_raw_outputs)
        inputs = flatten(self.validation_step_inputs)
        targets = flatten(self.validation_step_labels)

        self.validation_step_inputs.clear()
        self.validation_step_outputs.clear()
        self.validation_step_raw_outputs.clear()
        self.validation_step_labels.clear()

        conditioned_predictions = self.combine_conditioning_with_predictions(inputs, predictions)

        rewards = self.get_reward(conditioned_predictions)

        self.log("avg_reward_valid", np.mean(rewards))
        # TODO solve the val_check_interval issue
        # current_epoch_step = self.global_step % (self.current_epoch + 1)
        # logger.info(f"Epoch step: {current_epoch_step}")
        # logger.info(f"val_check_interval: {self.model_args['val_check_interval']}")
        # step_spacing = int(self.trainer.estimated_stepping_batches * self.model_args['val_check_interval'])
        # logger.info(f"Step spacing: {step_spacing}")
        if (
            self.model_args["save_validation_predictions"] and self.global_step % 10 == 0
        ):  # and current_epoch_step % step_spacing == 0 :
            logger.info(f"Global step: {self.global_step}")
            logger.info("Saving validation predictions to file.")
            logger.info(
                "Warning: this code does not consider the model version. Existing files will be overwritten."
            )
            output_dir = Path(self.model_args["output_dir"]) / "validation_outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            dump_list_to_file(predictions, output_dir / f"predictions-step{self.global_step}.txt")
            dump_list_to_file(
                raw_outputs, output_dir / f"raw-predictions-step{self.global_step}.txt"
            )
            dump_list_to_file(inputs, output_dir / f"inputs-step{self.global_step}.txt")
            dump_list_to_file(targets, output_dir / f"targets-step{self.global_step}.txt")
            dump_list_to_file(rewards, output_dir / f"raw-rewards-step{self.global_step}.txt")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["reference_model"] = (
            None if self.reference_model is None else self.reference_model.state_dict()
        )
        checkpoint["baseline_model"] = (
            None if self.baseline_model is None else self.baseline_model.model.state_dict()
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.reference_model is not None and checkpoint.get("reference_model", None) is not None:
            logger.info("Loading reference model state dict from checkpoint ...")
            self.reference_model.load_state_dict(checkpoint["reference_model"])
        if self.baseline_model is not None and checkpoint.get("baseline_model", None) is not None:
            logger.info("Loading baseline model state dict from checkpoint ...")
            self.baseline_model.model.load_state_dict(checkpoint["baseline_model"])


def flatten(line):
    return [item for sublist in line for item in sublist]
