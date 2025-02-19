"""Model utilities."""

from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.models.torch import device_claim, map_dict_sequences_to_tensors
from torch.nn import Softmax
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput


class ModelType(str, Enum):
    binary_classification = "classification"
    regression = "regression"
    mlm = "masked_language_modeling"
    fingerprint = "fingerprint"


MODEL_TYPE_TO_MODEL_CLASS = OrderedDict([
    (
        ModelType.binary_classification,
        AutoModelForSequenceClassification,
    ),  # uses the Cross Entropy loss
    (ModelType.regression, AutoModelForSequenceClassification),  # uses the RME loss
    (ModelType.mlm, AutoModelForMaskedLM),
    (ModelType.fingerprint, AutoModel),
])

MODEL_TYPE_TO_KWARGS = {
    ModelType.binary_classification: {"num_labels": 2},
    ModelType.regression: {"num_labels": 1},
    ModelType.mlm: {},
    ModelType.fingerprint: {},
}
RawModelOutput = Union[SequenceClassifierOutput, MaskedLMOutput, BaseModelOutput]
ModelOutput = Union[float, str, List[float], List[int]]


class RXNTransformersModelForReactions:
    def __init__(
        self,
        model_name_or_path: str,
        model_type: ModelType,
        tokenizer: SmilesTokenizer,
        maximum_length: int = 512,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Construct a RXNTransformersModel.
        Args:
            model_name_or_path: model name or path.
            model_type: model type.
            tokenizer: a tokenizer for reactions.
            maximum_length: maximum tokenized sequence length.
            device: device: device where the inference is running either as a dedicated class or
                a string. If not provided is inferred.
        Raises:
            ValueError: in case the model type is not supported.
        """
        self.model_name_or_path = model_name_or_path
        self.model_type = model_type
        if self.model_type in MODEL_TYPE_TO_MODEL_CLASS and self.model_type in MODEL_TYPE_TO_KWARGS:
            self.model = MODEL_TYPE_TO_MODEL_CLASS[self.model_type].from_pretrained(
                self.model_name_or_path, **MODEL_TYPE_TO_KWARGS[self.model_type]
            )
        else:
            raise ValueError(
                f"model_type={self.model_type} not supported! Select one from: {MODEL_TYPE_TO_MODEL_CLASS.keys()}"
            )
        self.tokenizer = tokenizer
        self.maximum_length = maximum_length
        self.device = device_claim(device)
        self.model.to(self.device)

    def tokenize_batch(self, rxns: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of reactions.
        Args:
            rxns: a list of reactions.
        Returns:
            a dictionary containing the token ids as well as token type ids and attention masks.
        """
        return map_dict_sequences_to_tensors(
            self.tokenizer.batch_encode_plus(
                rxns,
                add_special_tokens=True,
                max_length=self.maximum_length,
                pad_to_max_length=True,
                return_token_type_ids=True,
                return_tensors="pt",
            ),
            device=self.device,
        )

    def _postprocess_outputs(self, outputs: RawModelOutput) -> List[ModelOutput]:
        """Post-process raw model predictions to get the output.
        Args:
            outputs: raw model forward pass output.
        Raises:
            NotImplementedError: not implemented for base class RXNTransformersModelForReactions.
        Returns:
            the post-processed output.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement _postprocess_outputs!"
        )

    def predict(self, rxns: List[str]) -> List[ModelOutput]:
        """Run the model on a list of examples.
        Args:
            rxns: a list of reactions.
        Returns:
            a list of predictions.
        """
        with torch.no_grad():
            tokenized_batch = self.tokenize_batch(rxns)
            outputs = self.model(**tokenized_batch)
        return self._postprocess_outputs(outputs)


class RXNNegScorerModel(RXNTransformersModelForReactions):
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: SmilesTokenizer,
        maximum_length: int = 512,
        model_type: ModelType = ModelType.binary_classification,
    ) -> None:
        """Construct a RXNNegScorerModel.
        Args:
            model_name_or_path: model name or path.
            tokenizer: a tokenizer for reactions.
            maximum_length: maximum tokenized sequence length.
            model_type: the type of scorer, can be either a classifier or a regressor
        """
        if model_type not in [ModelType.binary_classification, ModelType.regression]:
            raise ValueError(
                f"Only the following model types are supported for this model:"
                f" {ModelType.binary_classification, ModelType.regression}"
            )

        super().__init__(
            model_name_or_path=model_name_or_path,
            model_type=model_type,
            tokenizer=tokenizer,
            maximum_length=maximum_length,
        )

    def _postprocess_outputs(self, outputs: RawModelOutput) -> Tuple[ModelOutput, ModelOutput]:
        """Post-process raw model predictions to get the output.
        Args:
            outputs: raw model forward pass output.
        Returns:
            the post-processed output.
        """
        if self.model_type == ModelType.binary_classification:
            sftmax = Softmax(dim=1)
            probabilities = sftmax(outputs.logits)
            predictions = torch.argmax(probabilities, dim=1).tolist()
            # at the end I want the probability to be positive
            return predictions, probabilities.tolist()
        else:
            # TODO: check
            # it is a regression
            # how do I get the probabilities? or should I keep the scores?
            predictions = [1 if p > 0.0 else 0 for p in outputs.logits.squeeze(1)]
            probabilities = [p + 0.5 for p in outputs.logits.squeeze(1)]
            return predictions, probabilities

    def generate_scores(self, batch: List[str]) -> List[float]:
        _, probabilities = self.predict(batch)
        # Return the probability of the positive class
        # TODO: adapt for regression case
        return [prob[1] for prob in probabilities]
