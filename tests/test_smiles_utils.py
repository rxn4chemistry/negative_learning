import random

import pandas as pd
from rxn_negative_learning.utils.smiles_utils import (
    oversample_reaction_minority_dataset,
    randomize_multiple_smiles_rotated,
)


def test_stable_randomization():
    random.seed(10)
    out1 = randomize_multiple_smiles_rotated("COc1ccc(Br)cn1")
    random.seed(10)
    out2 = randomize_multiple_smiles_rotated("COc1ccc(Br)cn1")
    assert out1 == out2


def test_oversample_reaction_minority_dataset():
    # Note that duplicates are not removed because otherwise I can augment some samples
    # more and some others less
    df = pd.DataFrame({
        "labels": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "text": [
            "Brc1cn[nH]c1",
            "Brc1cnn(-c2ncccn2)c1",
            "Cc1nn(-c2ccc(Cl)nn2)c(C)c1Cl",
            "Cn1cnc(Br)c1-c1ccccc1",
            "Brc1cnc(-c2nccs2)s1",
            "Nc1ncc(Br)c(-c2ccccc2)n1",
            "Nc1nnc(-c2ccccc2C(F)(F)F)cc1Br",
            "CCn1c(-c2nc(Br)cnc2N)nc2cnccc21",
            "COc1sc(-c2ccncc2)nc1Br",
            "COc1ccc(Br)cn1",
            "CNc1ncnc(N2CCOCC2)c1Br",
        ],
    })

    df = oversample_reaction_minority_dataset(df)
    print(df)
    assert len(df.loc[df["labels"] == 0]) == len(df.loc[df["labels"] == 1])
