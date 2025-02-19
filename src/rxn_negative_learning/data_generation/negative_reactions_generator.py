import logging
from typing import List, Optional

from rdkit import Chem, RDLogger
from rxn.chemutils.exceptions import InvalidSmiles
from rxn_negative_learning.data_generation.help_dicts_regiosqm import HALOGENS

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class RegioSQMdatum:
    def __init__(
        self,
        compound_name: str,
        main_reatant_smiles: str,
        reaction_centers: List[int],
        halo_reactants: List[str],
    ):
        self.compound_name = compound_name
        self.main_reactant_smiles = main_reatant_smiles
        self.main_reactant_mol = self._generate_mol_from_smiles(self.main_reactant_smiles)
        self.reaction_centers = reaction_centers
        self.halo_reactants = halo_reactants

    def _generate_mol_from_smiles(self, smiles: str):
        try:
            return Chem.MolFromSmiles(smiles)
        except InvalidSmiles:
            raise InvalidSmiles


class PosNegRxnGenerator:
    def __init__(self, regiosqmdatum: RegioSQMdatum):
        self.regiosqmdatum = regiosqmdatum

    def _generate(self):
        """
        Generates positives and negative reactions.
        """
        pos_reactions = []
        pos_targets = []
        neg_targets = []
        for halo_rectant in self.regiosqmdatum.halo_reactants:
            # generate a reaction where you add the halogen to the reactant
            first_halogen = True
            for halogen in HALOGENS:  # do not change order because of [Na]I.[Na+].[O-]Cl,
                # https://pubs.acs.org/doi/full/10.1021/acs.joc.8b02637.
                # What happens is that just the first halogen encountered is considered
                if halogen in halo_rectant:
                    if not first_halogen:
                        logging.info(
                            f"Not first halogen found in reactant {halo_rectant} for compound {self.regiosqmdatum.compound_name}"
                        )
                        break
                    # compute positive target
                    mol = self.regiosqmdatum.main_reactant_mol
                    original_number_of_atoms = len(mol.GetAtoms())
                    product = self.regiosqmdatum.main_reactant_smiles
                    for _ in self.regiosqmdatum.reaction_centers:
                        product += f".{halogen}"
                    edmol = Chem.EditableMol(Chem.MolFromSmiles(product))
                    for i, pos in enumerate(self.regiosqmdatum.reaction_centers):
                        edmol.AddBond(
                            pos,
                            original_number_of_atoms + i,
                            order=Chem.rdchem.BondType.SINGLE,
                        )

                    # generate alternative targets (also the positive will be generated)
                    alternative_targets_for_entry = self._generate_alternative_targets(halogen)

                    target = edmol.GetMol()
                    Chem.SanitizeMol(target)
                    pos_reactions.append(
                        ".".join([self.regiosqmdatum.main_reactant_smiles, halo_rectant])
                    )
                    target_smiles = Chem.MolToSmiles(target)
                    pos_targets.append(target_smiles)
                    # Remove alternative tgt equal to the positive, this happens for molecules like CN(C)c1ccncc1Br
                    neg_targets.append(
                        [neg for neg in alternative_targets_for_entry if neg != target_smiles]
                        if alternative_targets_for_entry is not None
                        else None
                    )
                    first_halogen = False
            if first_halogen:  # None halogen found
                logging.info(
                    f"No halogen found in reactant {halo_rectant} for compound {self.regiosqmdatum.compound_name}"
                )
        return pos_reactions, pos_targets, neg_targets

    def generate(self):
        return self._generate()

    def __call__(self):
        return self.generate()

    def _find_positions(self) -> List[int]:
        positions = []
        for i, atom in enumerate(self.regiosqmdatum.main_reactant_mol.GetAtoms()):
            if atom.GetIsAromatic() and atom.GetSymbol() == "C":
                positions.append(i)
        if not positions:
            logger.info(
                f"Sorry, no alternative positions could be found for smiles {self.regiosqmdatum.main_reactant_smiles}"
            )
        return positions

    def _generate_alternative_targets(self, halogen: str) -> Optional[List[str]]:
        positions = self._find_positions()
        if not positions:
            return None
        original_num_of_atoms = len(self.regiosqmdatum.main_reactant_mol.GetAtoms())
        alternative_targets = []
        for position in positions:
            neg_product = self.regiosqmdatum.main_reactant_smiles
            neg_product += f".{halogen}"
            edmol = Chem.EditableMol(Chem.MolFromSmiles(neg_product))
            edmol.AddBond(position, original_num_of_atoms, order=Chem.rdchem.BondType.SINGLE)
            target = edmol.GetMol()
            try:
                Chem.SanitizeMol(target)
            except Exception:
                logging.debug("Can't sanitize negative mol, skipping.")
                continue
            alternative_targets.append(Chem.MolToSmiles(target))
        negative_targets = sorted(list(set(alternative_targets)))
        if not negative_targets:
            logger.info(
                f"Could not generate alternative targets for reactant {self.regiosqmdatum.main_reactant_smiles}"
            )
            return None
        return negative_targets
