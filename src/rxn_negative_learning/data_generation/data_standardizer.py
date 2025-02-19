import logging
from abc import abstractmethod
from typing import TypeVar

from rxn.chemutils.conversion import canonicalize_smiles
from rxn.chemutils.exceptions import InvalidSmiles
from rxn.chemutils.reaction_equation import canonicalize_compounds, sort_compounds
from rxn.chemutils.reaction_smiles import parse_any_reaction_smiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound="TextDataStandardizer")


class TextDataStandardizer:
    def __init__(self, lazy: bool = False):
        self.lazy = lazy

    @abstractmethod
    def standardize_data(self, data_string: str):
        pass

    def standardize(self, data_string: str) -> str:
        """
        Convert a string to its standardized format
        """
        return self.standardize_data(data_string)


class SMILESDataStandardizer(TextDataStandardizer):
    def __init__(self, lazy: bool = True):
        super().__init__(lazy=lazy)

    def standardize_data(self, data_string: str):
        if ">>" in data_string:  # is a reaction
            try:
                return sort_compounds(
                    canonicalize_compounds(parse_any_reaction_smiles(data_string))
                ).to_string()
            except (InvalidSmiles, ValueError) as e:
                if self.lazy:
                    return ""
                else:
                    logger.exception(e)
                    raise ValueError(f"Unable to canonicalize rxn: {data_string}")

        else:  # is a molecule or set of molecules
            try:
                return canonicalize_smiles(data_string)
            except InvalidSmiles as e:
                if self.lazy:
                    return ""
                else:
                    logger.exception(e)
                    raise
