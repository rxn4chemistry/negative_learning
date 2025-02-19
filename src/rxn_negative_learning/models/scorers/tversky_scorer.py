from typing import List, Optional

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from rxn_negative_learning.models.scorers.ideal_scorer import IdealScorer
from rxn_negative_learning.utils.smiles_utils import (
    get_precursors_from_reactions,
    get_precursors_from_smiles,
    get_product_from_reactions,
)


class TverskyScorer(IdealScorer):
    def __init__(
        self,
        positive_reactions: List[str],
        negative_reactions: List[str] = None,
        shift: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        fps_type: Optional[str] = None,
        radius: int = 2,
    ):
        """
        Here it is better to not include the negative reactions to avoid grounding in order to use tanimoto
        sui negatives
        """

        super().__init__(positive_reactions, negative_reactions, shift)
        self.src2tgt_pos = dict(
            zip(
                get_precursors_from_reactions(positive_reactions),
                get_product_from_reactions(positive_reactions),
            )
        )
        self.alpha = alpha
        self.beta = beta
        self.fps_type = fps_type
        self.radius = radius

    def score(self, reactions: List[str]) -> List[float]:
        labels = [
            self.src2tgt_pos.get(get_precursors_from_smiles(reaction), "") for reaction in reactions
        ]
        return [
            self.tversky_score(reaction, label) if (reaction != "" and label != "") else 0.0
            for reaction, label in zip(reactions, labels)
        ]

    def tversky_score(self, smiles1, smiles2):
        # Act on the product
        if self.fps_type == "morgan":
            fpgen = GetMorganGenerator(radius=self.radius)
        else:
            fpgen = GetRDKitFPGenerator()
        mols = [
            Chem.MolFromSmiles(smiles1),
            Chem.MolFromSmiles(smiles2),
        ]
        fps = [fpgen.GetFingerprint(x) for x in mols]
        return DataStructs.TverskySimilarity(fps[0], fps[1], self.alpha, self.beta)
