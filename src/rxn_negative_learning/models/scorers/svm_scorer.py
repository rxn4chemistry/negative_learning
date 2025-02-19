import logging
import pickle
from pathlib import Path
from typing import List, Optional

import torch
from rxn_negative_learning.models.scorers.scorer_base import RXNNegScorerBase
from rxn_negative_learning.models.tokenization import SmilesTokenizer
from rxn_negative_learning.scripts.finetune_scorer import get_associated_max_len
from rxn_negative_learning.utils.repo_utils import models_directory
from rxn_negative_learning.utils.smiles_utils import get_precursors_from_smiles
from transformers import AutoModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# # Hacky but the dumping of this model is problematic
# EMBEDDINGS_MODEL_PATH = models_directory() / "old" / "svm_scorer"
# logger.info(f"Embeddings model path: {EMBEDDINGS_MODEL_PATH}")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# EMBEDDINGS_MODEL = AutoModel.from_pretrained(
#     pretrained_model_name_or_path=EMBEDDINGS_MODEL_PATH
# ).to(DEVICE)


class SVMScorer(RXNNegScorerBase):
    def __init__(
        self,
        model_path: Optional[Path] = None,
        positive_reactions: Optional[List[str]] = None,
        negative_reactions: Optional[List[str]] = None,
        shift: float = 0.0,
    ):
        """
        Here the positive and negative reactions define the grounding, but the scores are computed
        by a svm model on pretrained embeddings
        """
        super().__init__(positive_reactions, negative_reactions, shift)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Initializing embeddings model ...")
        self.embeddings_model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_path
        ).to(self.device)
        self.embeddings_model_max_length = get_associated_max_len(model_path)
        logger.info(f"Embeddings model max sequence length: {self.embeddings_model_max_length}")

        logger.info("Initializing embeddings model tokenizer ...")
        self.tokenizer = SmilesTokenizer.from_pretrained(model_path)

        logger.info("Loading svm model ...")
        self.svm_model_file = model_path / "svm_model.pkl"
        if not self.svm_model_file.exists():
            raise RuntimeError(f"No svm model file found where expected: {self.svm_model_file}")
        with open(self.svm_model_file, "rb") as f:
            self.svm_model = pickle.load(f)

    def score(self, reactions: List[str]) -> List[float]:
        tokenized_reactions = self.tokenizer(
            reactions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.embeddings_model_max_length,
        ).to(self.device)
        with torch.no_grad():
            hidden = self.embeddings_model(**tokenized_reactions)
            embeddings = hidden.last_hidden_state[:, 0, :].cpu().numpy().tolist()
            scores = self.svm_model.predict(embeddings)
        return [
            sc if get_precursors_from_smiles(rxn) != "" else 0.0
            for sc, rxn in zip(scores, reactions)
        ]
