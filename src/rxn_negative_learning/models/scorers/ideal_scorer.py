import logging
from typing import List, Optional

from rxn_negative_learning.models.scorers.scorer_base import RXNNegScorerBase

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class IdealScorer(RXNNegScorerBase):
    def __init__(
        self,
        positive_reactions: Optional[List[str]] = None,
        negative_reactions: Optional[List[str]] = None,
        shift: float = 0.0,
    ):
        super().__init__(positive_reactions, negative_reactions, shift)
        if positive_reactions is None:
            logger.info("Attention, no positive reactions provided!")

    def score(self, reactions: List[str]) -> List[float]:
        return [0.0 for _ in range(len(reactions))]
