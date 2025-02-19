from typing import List

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator
from rxn_negative_learning.models.scorers.scorer_base import RXNNegScorerBase
from rxn_negative_learning.utils.smiles_utils import (
    get_precursors_from_reactions,
    get_precursors_from_smiles,
    get_product_from_reactions,
    get_product_from_smiles,
)


class TanimotoIdealScorer(RXNNegScorerBase):
    def __init__(
        self,
        positive_reactions: List[str],
        negative_reactions: List[str] = None,
        shift: float = 0.0,
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

    def score(self, reactions: List[str]) -> List[float]:
        predictions = [get_product_from_smiles(reaction) for reaction in reactions]
        labels = [
            self.src2tgt_pos.get(get_precursors_from_smiles(reaction), "") for reaction in reactions
        ]
        return [
            self.tanimoto_score(pred, label) if (pred != "" and label != "") else 0.0
            for pred, label in zip(predictions, labels)
        ]

    def tanimoto_score(self, smiles1, smiles2):
        # Act on the product
        fpgen = GetRDKitFPGenerator()
        mols = [
            Chem.MolFromSmiles(smiles1),
            Chem.MolFromSmiles(smiles2),
        ]
        fps = [fpgen.GetFingerprint(x) for x in mols]
        return DataStructs.TanimotoSimilarity(fps[0], fps[1])
