from abc import abstractmethod
from typing import List, Optional


class RXNNegScorerBase:
    def __init__(
        self,
        positive_reactions: Optional[List[str]] = None,
        negative_reactions: Optional[List[str]] = None,
        shift: float = 0.0,
    ):
        self.shift = shift
        self.positive_reactions = positive_reactions if positive_reactions is not None else []
        self.negative_reactions = negative_reactions if negative_reactions is not None else []

    def __call__(self, reactions: List[str]):
        return self.get_reward(reactions)

    def get_reward(self, reactions: List[str]):
        scores = self.score(reactions)
        grounded_scores = self.ground(scores, reactions)
        return self.apply_shift(grounded_scores)

    @abstractmethod
    def score(self, reactions: List[str]) -> List[float]:
        pass

    def apply_shift(self, scores: List[float]) -> List[float]:
        return [(sc - self.shift) for sc in scores]

    def ground(self, scores: List[float], reactions: List[str]):
        grounded_scores = [
            scores[i] if reactions[i] not in self.positive_reactions else 1.0
            for i in range(len(reactions))
        ]
        grounded_scores = [
            grounded_scores[i] if reactions[i] not in self.negative_reactions else 0.0
            for i in range(len(reactions))
        ]
        return grounded_scores
