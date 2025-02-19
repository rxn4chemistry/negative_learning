from typing import List

from Levenshtein import distance
from rxn_negative_learning.models.scorers.scorer_base import RXNNegScorerBase
from rxn_negative_learning.utils.smiles_utils import (
    get_precursors_from_reactions,
    get_precursors_from_smiles,
    get_product_from_reactions,
    get_product_from_smiles,
)


class LevenshteinIdealScorer(RXNNegScorerBase):
    def __init__(
        self,
        positive_reactions: List[str],
        negative_reactions: List[str] = None,
        shift: float = 0.0,
    ):
        """
        Here it is better to not include the negative reactions to avoid grounding in order to use levenshtein
        sui negatives
        """
        super().__init__(positive_reactions, negative_reactions, shift)
        self.src2tgt_pos = dict(
            zip(
                get_precursors_from_reactions(positive_reactions),
                get_product_from_reactions(positive_reactions),
            )
        )

    def score(self, reactions: List[str]) -> List[float]:
        predictions = [get_product_from_smiles(reaction) for reaction in reactions]
        labels = [
            self.src2tgt_pos.get(get_precursors_from_smiles(reaction), "") for reaction in reactions
        ]
        return [
            self.levenshtein_score(pred, label) if (pred != "" and label != "") else 0.0
            for pred, label in zip(predictions, labels)
        ]

    def levenshtein_score(self, string1, string2):
        return 1 / (1 + distance(string1, string2))
