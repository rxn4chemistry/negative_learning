import json
import random
from enum import Enum, auto
from pathlib import Path
from typing import Any, List

import pandas as pd
from rdkit import Chem
from rxn.chemutils import smiles_randomization
from rxn.chemutils.conversion import canonicalize_smiles, mol_to_smiles, smiles_to_mol
from rxn.chemutils.exceptions import InvalidSmiles


class RandomType(Enum):
    molecules = auto()
    unrestricted = auto()
    restricted = auto()
    rotated = auto()
    combined = auto()


def partial_smiles_sequences(tok_smi: str):
    smi_list = tok_smi.split(" ")
    for i in range(1, len(smi_list) + 1):
        yield "".join(smi_list[0:i])


def lazy_canonicalize_smiles(smile: str):
    try:
        return canonicalize_smiles(smile)
    except InvalidSmiles:
        return ""


def flatten(double_list: List[List[Any]]):
    return [item for sublist in double_list for item in sublist]


def get_product_from_smiles(rxn: str) -> str:
    return rxn.split(">>")[-1]


def get_precursors_from_smiles(rxn: str) -> str:
    return rxn.split(">>")[0]


def get_precursors_from_reactions(rxns: List[str]) -> List[str]:
    return [get_precursors_from_smiles(smi) for smi in rxns]


def get_product_from_reactions(rxns: List[str]) -> List[str]:
    return [get_product_from_smiles(smi) for smi in rxns]


def get_reaction_from_precursors_and_product_smiles(precursors: str, product: str) -> str:
    return f"{precursors}>>{product}"


def randomize_smiles_restricted(smiles: str, seed: int) -> str:
    """
    Randomize a SMILES string in a restricted fashion.
    Raises:
        InvalidSmiles: for invalid molecules.
    Args:
        smiles: SMILES string to randomize.
        seed: an int to stabilize the random generation, for reproducibility.
    Returns:
        Randomized SMILES string.
    """
    mol = smiles_to_mol(smiles, sanitize=False)
    new_atom_order = list(range(mol.GetNumAtoms()))
    random.seed(seed)
    random.shuffle(new_atom_order)
    mol = Chem.RenumberAtoms(mol, newOrder=new_atom_order)
    return mol_to_smiles(mol, canonical=False)


def randomize_multiple_smiles_rotated(smiles: str):
    randomized_elems = []
    for elem in smiles.split(">>"):
        smiles_list = elem.split(".")
        randomized_list = [
            smiles_randomization.randomize_smiles_rotated(smi) for smi in smiles_list
        ]
        randomized_elems.append(".".join(randomized_list))
    return ">>".join(randomized_elems)


def randomize_multiple_smiles_restricted(smiles: str):
    randomized_elems = []
    for elem in smiles.split(">>"):
        smiles_list = elem.split(".")
        randomized_list = [
            smiles_randomization.randomize_smiles_restricted(smi) for smi in smiles_list
        ]
        randomized_elems.append(".".join(randomized_list))
    return ">>".join(randomized_elems)


def randomize_multiple_smiles_unrestricted(smiles: str):
    randomized_elems = []
    for elem in smiles.split(">>"):
        smiles_list = elem.split(".")
        randomized_list = [
            smiles_randomization.randomize_smiles_unrestricted(smi) for smi in smiles_list
        ]
        randomized_elems.append(".".join(randomized_list))
    return ">>".join(randomized_elems)


def randomize_smiles_rotated(smiles: str, seed: int, with_order_reversal: bool = True) -> str:
    """
    Randomize a SMILES string by doing a cyclic rotation of the atomic indices.
    Adapted from https://github.com/GLambard/SMILES-X/blob/758478663030580a363a9ee61c11f6d6448e18a1/SMILESX/augm.py#L19.
    Raises:
        InvalidSmiles: for invalid molecules.
    Args:
        smiles: SMILES string to randomize.
        seed: an int to stabilize the random generation, for reproducibility.
        with_order_reversal: whether to reverse the atom order with 50% chance.
    Returns:
        Randomized SMILES string.
    """

    mol = smiles_to_mol(smiles, sanitize=False)

    n_atoms = mol.GetNumAtoms()

    # Generate random values
    random.seed(seed)
    rotation_index = random.randint(0, n_atoms - 1)
    random.seed(seed)
    reverse_order = with_order_reversal and random.choice([True, False])

    # Generate new atom indices order
    atoms = list(range(n_atoms))
    new_atoms_order = atoms[rotation_index % len(atoms) :] + atoms[: rotation_index % len(atoms)]
    if reverse_order:
        new_atoms_order.reverse()

    mol = Chem.RenumberAtoms(mol, new_atoms_order)
    return mol_to_smiles(mol, canonical=False)


def randomize_smiles_unrestricted(smiles: str, seed: int) -> str:
    """
    Randomize a SMILES string in an unrestricted fashion.
    Raises:
        InvalidSmiles: for invalid molecules.
    Args:
        smiles: SMILES string to randomize.
        seed: an int to stabilize the random generation, for reproducibility.
    Returns:
        Randomized SMILES string.
    """
    raise ValueError("Currently non stable!")
    # mol = smiles_to_mol(smiles, sanitize=False)
    # return Chem.MolToSmiles(mol, canonical=False, doRandom=True)


def load_jsonl_dataset_to_dataframe(jsonl_dataset_file: Path) -> pd.DataFrame:
    sources = []
    targets = []
    opp_targets = []
    scores = []
    ids = []
    with open(jsonl_dataset_file, "r") as f:
        data = [json.loads(line.strip()) for line in f]
        for elem in data:
            sources.append(elem["source"])
            targets.append(elem["target"])
            opp_targets.append(elem["opposite_targets"])
            scores.append(elem["score"])
            ids.append(elem["idx"])
    return pd.DataFrame({
        "source": sources,
        "target": targets,
        "opposite_targets": opp_targets,
        "score": scores,
        "idx": ids,
    })


def oversample_reaction_minority_dataset(
    df: pd.DataFrame, label_column: str = "labels", target_column: str = "text"
):
    """
    Function to create a dataset where the minority class is oversampled.
    The instances from the minority class are augmented with SMILES
    rotation until they reach the amount of the majority class.
    Using the unrestricted randomization to give more diversity
    """

    labels_frequency_dict = dict(df[label_column].value_counts())
    majority_label = max(labels_frequency_dict, key=labels_frequency_dict.get)
    minority_lab = min(labels_frequency_dict, key=labels_frequency_dict.get)
    minority_df = df.loc[df[label_column] == minority_lab]
    random.seed(42)
    new_df = minority_df.copy()
    new_length = len(minority_df)
    while new_length < labels_frequency_dict[majority_label]:
        # This augmentation should change at every iter
        random.seed(new_length**2 - 1)
        minority_df["augmented"] = minority_df.apply(
            lambda x: randomize_multiple_smiles_restricted(x[target_column]), axis=1
        )

        if len(new_df) + len(minority_df) < labels_frequency_dict[majority_label]:
            new_df = pd.concat([
                new_df[[target_column, label_column]],
                minority_df[["augmented", label_column]].rename(
                    columns={"augmented": target_column}
                ),
            ])
        else:
            n_sample = len(new_df) + len(minority_df) - labels_frequency_dict[majority_label]
            new_df = pd.concat([
                new_df[[target_column, label_column]],
                minority_df[["augmented", label_column]]
                .rename(columns={"augmented": target_column})
                .sample(n=n_sample, random_state=42),
            ])
        # minority_df.drop_duplicates(subset=[target_column], inplace=True)
        new_length = len(new_df)

    df = pd.concat([df.loc[df[label_column] == majority_label], new_df])
    return df.sample(n=len(df), random_state=42).reset_index(drop=True)
