""" A utility class to augment the dataset files """

import math
import random
from typing import Iterable, List, Tuple

import pandas as pd
from rxn.chemutils.tokenization import detokenize_smiles, tokenize_smiles
from rxn.utilities.files import PathLike, load_list_from_file
from rxn_negative_learning.utils import (
    RandomType,
    randomize_smiles_restricted,
    randomize_smiles_rotated,
    randomize_smiles_unrestricted,
)


def molecules_permutation_given_index(
    molecules_list: List[str], permutation_index: int
) -> List["str"]:
    """
    https://stackoverflow.com/questions/5602488/random-picks-from-permutation-generator
    """
    molecules_list = molecules_list[:]
    for i in range(len(molecules_list) - 1):
        permutation_index, j = divmod(permutation_index, len(molecules_list) - i)
        molecules_list[i], molecules_list[i + j] = (
            molecules_list[i + j],
            molecules_list[i],
        )
    return molecules_list


class Augmenter:
    def __init__(
        self,
        df: pd.DataFrame,
        src_column_name: str,
        tgt_column_name: str,
        fragment_bond: str = ".",
        seed: int = 42,
    ):
        """Creates a new instance of the Augmenter class.
        Args:
            df (pd.DataFrame): A pandas DataFrame containing the molecules SMILES.
            src_column_name: The name of the DataFrame column containing the source reaction SMILES.
            tgt_column_name: The name of the DataFrame column containing the target.
            fragment_bond (str): The fragment bond token contained in the SMILES.
        """
        self.df = df
        self.__src_column_name = src_column_name
        self.__tgt_column_name = tgt_column_name
        self.tokenizer = tokenize_smiles
        self.fragment_bond = fragment_bond
        self.seed = seed

    #
    # Private Methods
    #

    def __randomize_smiles(
        self, smiles: str, random_type: RandomType, permutations: int
    ) -> List[str]:
        """
        Randomizes a molecules SMILES string that might contain fragment bonds
        and returns a number of augmented versions of the SMILES equal to permutations.
        Args:
            smiles (str): The molecules SMILES to augment
            random_type (RandomType): The type of randomization to be applied.
            permutations (int): The number of permutations to deliver for the SMILES
        Returns:
            List[str]: The list of randomized SMILES
        """
        # Raise for empty SMILES
        if not smiles:
            raise ValueError

        list_of_smiles: List[str] = []
        for i in range(permutations):
            list_of_smiles.append(
                ".".join([
                    self.fragment_bond.join([
                        Augmenter.__randomize_smiles_without_fragment(
                            fragment, random_type, seed=self.seed + i
                        )
                        for fragment in group.split(self.fragment_bond)
                    ])
                    for group in smiles.split(".")
                ])
            )
        return list_of_smiles

    #
    # Private Static Methods
    #

    @staticmethod
    def __randomize_smiles_without_fragment(smiles: str, random_type: RandomType, seed: int) -> str:
        """
        Generates a random version of a SMILES without a fragment bond
        Args:
            smiles (str): The pandas DataFrame to be split into training, validation, and test sets.
            random_type (RandomType): The type of randomization to be applied.
        Raises:
            InvalidSmiles: for invalid SMILES (raised via rxn_chemutils).
            ValueError: if an invalid randomization type is provided.
        Returns:
            str: the randomized SMILES
        """
        if random_type == RandomType.unrestricted:
            return randomize_smiles_unrestricted(smiles, seed=seed)
        elif random_type == RandomType.restricted:
            return randomize_smiles_restricted(smiles, seed=seed)
        elif random_type == RandomType.rotated:
            return randomize_smiles_rotated(smiles, seed=seed, with_order_reversal=True)
        raise ValueError(f"Invalid random type: {random_type}")

    def __randomize_molecules(self, smiles: str, permutations: int) -> List[str]:
        """
        Randomizes the order of the molecules inside a SMILES string that might
        contain fragment bonds and returns a number of augmented versions of the
        SMILES equal to permutations.
        For a number of molecules smaller than permutations, returns a number of
        permutations equal to the number of molecules
        Args:
            smiles (str): The molecules SMILES to augment
            permutations (int): The number of permutations to deliver for the SMILES
        Returns:
            List[str]: The list of randomized SMILES
        """
        # Raise for empty SMILES
        if not smiles:
            raise ValueError

        molecules_list = smiles.split(".")
        total_permutations = range(min(math.factorial(len(molecules_list)), 4000000))
        random.seed(self.seed)
        permutation_indices = random.sample(
            total_permutations, min(permutations, len(molecules_list))
        )
        permuted_molecules_smiles = []
        for idx in permutation_indices:
            permuted_precursors = molecules_permutation_given_index(molecules_list, idx)
            permuted_molecules_smiles.append(".".join(permuted_precursors))

        return permuted_molecules_smiles

    #
    # Public Methods
    #

    def augment(
        self,
        random_type: RandomType = RandomType.rotated,
        permutations: int = 1,
    ) -> Tuple[Iterable[str], Iterable[str]]:
        """
        Creates samples for the augmentation. Returns a a pandas Series containing the
        augmented samples.
        Args:
            random_type (RandomType): The string identifying the type of randomization to apply.
                "molecules" for randomization of the molecules (canonical SMILES kept)
                "unrestricted" for unrestricted randomization
                "restricted" for restricted randomization
                "rotated" for rotated randomization
                For details on the differences:
                https://github.com/undeadpixel/reinvent-randomized and
                https://github.com/GLambard/SMILES-X
            permutations (int): The number of permutations to generate for each SMILES
        Returns:
            pd.DataFrame: A pandas Series containing the augmented samples.
        """

        self.df[f"{self.__src_column_name}_{random_type.name}"] = self.df[
            f"{self.__src_column_name}"
        ].apply(lambda smiles: detokenize_smiles(smiles))
        column_to_augment = f"{self.__src_column_name}_{random_type.name}"

        if random_type == RandomType.molecules:
            self.df[column_to_augment] = self.df[column_to_augment].apply(
                lambda smiles: self.__randomize_molecules(smiles, permutations)
            )
        elif random_type == RandomType.combined:

            def combine_randomizations(
                rotated_list: List[str],
                restricted_list: List[str],
                molorder_list: List[str],
            ):
                full_list = rotated_list.copy()
                full_list.extend(restricted_list)
                full_list.extend(molorder_list)
                if len(full_list) != 3:
                    print("Hello")
                return full_list

            self.df[column_to_augment] = self.df[column_to_augment].apply(
                lambda smiles: combine_randomizations(
                    self.__randomize_smiles(smiles, RandomType.rotated, permutations),
                    self.__randomize_smiles(smiles, RandomType.restricted, permutations),
                    self.__randomize_molecules(smiles, permutations),
                )
            )
        else:
            self.df[column_to_augment] = self.df[column_to_augment].apply(
                lambda smiles: self.__randomize_smiles(smiles, random_type, permutations)
            )

        # Exploding the dataframe columns where I have the list of augmented
        # versions of a SMILES (the list length is the number of permutations)
        self.df = (
            self.df.set_index([col for col in self.df.keys() if col != column_to_augment])
            .apply(pd.Series.explode)
            .reset_index()
        )

        # tokenize
        self.df[column_to_augment] = self.df[column_to_augment].apply(lambda x: tokenize_smiles(x))

        # Joining the canonical and the augmented reactions
        augmented_df = pd.concat([
            self.df[[self.__src_column_name, self.__tgt_column_name]].drop_duplicates(),
            self.df[
                [
                    f"{self.__src_column_name}_{random_type.name}",
                    self.__tgt_column_name,
                ]
            ].rename(
                columns={f"{self.__src_column_name}_{random_type.name}": self.__src_column_name}
            ),
        ])

        # shuffle so canonical and augmented versions are mixed
        augmented_df = augmented_df.sample(n=len(augmented_df), random_state=self.seed)

        return (
            augmented_df[self.__src_column_name].values,
            augmented_df[self.__tgt_column_name].values,
        )

    #
    # Public Static Methods
    #

    @staticmethod
    def read_txts(
        src_filepath: PathLike, tgt_filepath: PathLike, fragment_bond: str = "."
    ) -> "Augmenter":
        """A helper function to read a list of SMILES and their target.
        Args:
            src_filepath: The path to the text file containing the molecules SMILES.
            tgt_filepath: The path to the text file containing the target.
            fragment_bond (str): The fragment token in the reaction SMILES
        Returns:
            Augmenter: A new augmenter instance.
        """
        src = load_list_from_file(src_filepath)
        tgt = load_list_from_file(tgt_filepath)

        return Augmenter(
            df=pd.DataFrame({"src": src, "tgt": tgt}),
            src_column_name="src",
            tgt_column_name="tgt",
            fragment_bond=fragment_bond,
        )
