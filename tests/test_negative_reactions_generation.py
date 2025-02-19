from ast import literal_eval

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rxn_negative_learning.data_generation.negative_reactions_generator import (
    PosNegRxnGenerator,
    RegioSQMdatum,
)
from rxn_negative_learning.scripts.generate_buitrago_dataset import generate_buitrago_negatives
from rxn_negative_learning.scripts.generate_regiosqm_dataset import generate_regiosqm_negatives
from rxn_negative_learning.scripts.generate_uspto_dataset import process_uspto_dataset

enumerator = rdMolStandardize.TautomerEnumerator()


def test_regiosqm_datum_creation():
    data = {
        1: {
            "compound_name": "comp33",
            "main_reatant_smiles": "c1cccc(c1C(F)(F)F)c1nnc(cc1)N",
            "reaction_centers": [14],
            "halo_reactants": ["BrBr.[Na+].OC([O-])=O"],
        }
    }
    datum = RegioSQMdatum(
        data[1]["compound_name"],
        data[1]["main_reatant_smiles"],
        data[1]["reaction_centers"],
        data[1]["halo_reactants"],
    )

    assert Chem.MolToSmiles(datum.main_reactant_mol) == Chem.MolToSmiles(
        Chem.MolFromSmiles("c1cccc(c1C(F)(F)F)c1nnc(cc1)N")
    )


def test_pos_neg_rxn_generator():
    data = {
        1: {
            "compound_name": "comp33",
            "main_reatant_smiles": "c1cccc(c1C(F)(F)F)c1nnc(cc1)N",
            "reaction_centers": [14],
            "halo_reactants": ["BrBr.[Na+].OC([O-])=O"],
        },
        2: {
            "compound_name": "comp154",
            "main_reatant_smiles": "c1cn(c(n1)C)C",
            "reaction_centers": [1, 0],
            "halo_reactants": ["O=C1N(Br)C(=O)CC1"],
        },
        3: {
            "compound_name": "comp130",
            "main_reatant_smiles": "c1ccc(cn1)O",
            "reaction_centers": [0],
            "halo_reactants": ["[Na]I.[Na+].[O-]Cl"],
        },
        4: {
            "compound_name": "comp163",
            "main_reatant_smiles": "c1cn(cn1)C(c1ccccc1)(c1ccccc1Cl)c1ccccc1",
            "reaction_centers": [0],
            "halo_reactants": ["O=C1N(Br)C(=O)CC1", "O=C1N(Cl)C(=O)CC1"],
        },
    }
    # datum with one reaction centre and one halogen
    datum = RegioSQMdatum(
        data[1]["compound_name"],
        data[1]["main_reatant_smiles"],
        data[1]["reaction_centers"],
        data[1]["halo_reactants"],
    )
    pos_neg_generator = PosNegRxnGenerator(datum)
    pos_reactions, pos_targets, neg_targets = pos_neg_generator()

    assert pos_reactions == ["c1cccc(c1C(F)(F)F)c1nnc(cc1)N.BrBr.[Na+].OC([O-])=O"]
    assert pos_targets == ["Nc1nnc(-c2ccccc2C(F)(F)F)cc1Br"]
    assert neg_targets == [
        [
            "Nc1cc(Br)c(-c2ccccc2C(F)(F)F)nn1",
            "Nc1ccc(-c2c(Br)cccc2C(F)(F)F)nn1",
            "Nc1ccc(-c2cc(Br)ccc2C(F)(F)F)nn1",
            "Nc1ccc(-c2ccc(Br)cc2C(F)(F)F)nn1",
            "Nc1ccc(-c2cccc(Br)c2C(F)(F)F)nn1",
        ]
    ]

    # datum with two reaction centers and one halo reactant
    datum = RegioSQMdatum(
        data[2]["compound_name"],
        data[2]["main_reatant_smiles"],
        data[2]["reaction_centers"],
        data[2]["halo_reactants"],
    )
    pos_neg_generator = PosNegRxnGenerator(datum)
    pos_reactions, pos_targets, neg_targets = pos_neg_generator()

    assert pos_reactions == ["c1cn(c(n1)C)C.O=C1N(Br)C(=O)CC1"]
    assert pos_targets == ["Cc1nc(Br)c(Br)n1C"]
    assert neg_targets == [["Cc1nc(Br)cn1C", "Cc1ncc(Br)n1C"]]

    # datum with one reaction centre and two halogens
    datum = RegioSQMdatum(
        data[3]["compound_name"],
        data[3]["main_reatant_smiles"],
        data[3]["reaction_centers"],
        data[3]["halo_reactants"],
    )
    pos_neg_generator = PosNegRxnGenerator(datum)
    pos_reactions, pos_targets, neg_targets = pos_neg_generator()

    assert pos_reactions == ["c1ccc(cn1)O.[Na]I.[Na+].[O-]Cl"]
    assert pos_targets == ["Oc1ccc(I)nc1"]
    assert neg_targets == [["Oc1cccnc1I", "Oc1cncc(I)c1", "Oc1cnccc1I"]]

    # datum with one reaction centre and two halo reactant
    datum = RegioSQMdatum(
        data[4]["compound_name"],
        data[4]["main_reatant_smiles"],
        data[4]["reaction_centers"],
        data[4]["halo_reactants"],
    )
    pos_neg_generator = PosNegRxnGenerator(datum)
    pos_reactions, pos_targets, neg_targets = pos_neg_generator()

    assert pos_reactions == [
        "c1cn(cn1)C(c1ccccc1)(c1ccccc1Cl)c1ccccc1.O=C1N(Br)C(=O)CC1",
        "c1cn(cn1)C(c1ccccc1)(c1ccccc1Cl)c1ccccc1.O=C1N(Cl)C(=O)CC1",
    ]
    assert pos_targets == [
        "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1cnc(Br)c1",
        "Clc1cn(C(c2ccccc2)(c2ccccc2)c2ccccc2Cl)cn1",
    ]
    print(neg_targets)
    assert neg_targets == [
        [
            "Clc1c(Br)cccc1C(c1ccccc1)(c1ccccc1)n1ccnc1",
            "Clc1cc(Br)ccc1C(c1ccccc1)(c1ccccc1)n1ccnc1",
            "Clc1ccc(Br)cc1C(c1ccccc1)(c1ccccc1)n1ccnc1",
            "Clc1cccc(Br)c1C(c1ccccc1)(c1ccccc1)n1ccnc1",
            "Clc1ccccc1C(c1ccccc1)(c1ccc(Br)cc1)n1ccnc1",
            "Clc1ccccc1C(c1ccccc1)(c1cccc(Br)c1)n1ccnc1",
            "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1Br",
            "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1cncc1Br",
            "Clc1ccccc1C(c1ccccc1)(c1ccccc1Br)n1ccnc1",
        ],
        [
            "Clc1ccc(C(c2ccccc2)(c2ccccc2)n2ccnc2)c(Cl)c1",
            "Clc1ccc(C(c2ccccc2)(c2ccccc2Cl)n2ccnc2)cc1",
            "Clc1ccc(Cl)c(C(c2ccccc2)(c2ccccc2)n2ccnc2)c1",
            "Clc1cccc(C(c2ccccc2)(c2ccccc2)n2ccnc2)c1Cl",
            "Clc1cccc(C(c2ccccc2)(c2ccccc2Cl)n2ccnc2)c1",
            "Clc1cccc(Cl)c1C(c1ccccc1)(c1ccccc1)n1ccnc1",
            "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1ccnc1Cl",
            "Clc1ccccc1C(c1ccccc1)(c1ccccc1)n1cncc1Cl",
            "Clc1ccccc1C(c1ccccc1)(c1ccccc1Cl)n1ccnc1",
        ],
    ]


def test_regiosqm_dataset_generation(tmp_path):
    regiosqm_data = pd.DataFrame(
        [
            ["comp1", "n1ccc[nH]1", [2], ["O=C1N(Br)C(=O)CC1"]],
            ["comp2", "n1cccn1c1ncccn1", [2], ["BrBr"]],
            ["comp3", "n1ccc(n1c1c(cccc1)C)N", [2], ["BrBr"]],
        ],
        columns=["name", "main_reactant", "reaction_centers", "halo_reactants"],
    )
    d = tmp_path / "regiosqm_data"
    d.mkdir()
    in_f = d / "tmp.txt"
    out_f = d / "tmp_out.txt"
    regiosqm_data.to_csv(in_f, index=False)

    reload_regiosqm_data = pd.read_csv(in_f)
    reload_regiosqm_data["reaction_centers"] = reload_regiosqm_data["reaction_centers"].apply(
        lambda x: literal_eval(x)
    )
    reload_regiosqm_data["halo_reactants"] = reload_regiosqm_data["halo_reactants"].apply(
        lambda x: literal_eval(x)
    )
    assert (regiosqm_data == reload_regiosqm_data).all().all()

    generate_regiosqm_negatives(input_csv=in_f, output_csv=out_f, keep_no_negatives=False)
    out_regiosqm_data = pd.read_csv(out_f)
    assert out_regiosqm_data["rxn"].tolist()[0:3] == [
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cc[nH]n1",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1ccn[nH]1",
    ]
    assert out_regiosqm_data["score"].tolist()[0:3] == [1, 0, 0]
    assert out_regiosqm_data["idx"].tolist()[0:3] == [0, 0, 0]


def test_buitrago_dataset_generation(tmp_path):
    buitrago_data = pd.DataFrame(
        [
            [
                "CS(=O)C.Brc1cccnc1.CC(N)CCc1ccccc1.C1CCC2=NCCCN2CC1.CS(=O)(=O)O[Pd-]1[NH2+]c2ccccc2-c2ccccc21.c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1>>CC(CCc1ccccc1)Nc1cccnc1",
                0.1,
            ],
            [
                "CS(=O)C.Brc1cccnc1.CC(N)CCc1ccccc1.CN1CCCN2CCCN=C12.CS(=O)(=O)O[Pd-]1[NH2+]c2ccccc2-c2ccccc21.c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1>>CC(CCc1ccccc1)Nc1cccnc1",
                0.0,
            ],
        ],
        columns=["rxn_smiles", "Pd/IS"],
    )
    d = tmp_path / "buitrago_data"
    d.mkdir()
    in_f = d / "tmp.txt"
    out_f = d / "tmp_out.txt"
    buitrago_data.to_csv(in_f, index=False)

    reload_buitrago_data = pd.read_csv(in_f)
    assert (buitrago_data == reload_buitrago_data).all().all()

    generate_buitrago_negatives(input_csv=in_f, output_csv=out_f, threshold=0.0)
    out_buitrago_data = pd.read_csv(out_f)
    assert out_buitrago_data["score"].tolist()[0:3] == [1, 0]


def test_uspto_dataset_generation(tmp_path):
    uspto_data = pd.DataFrame(
        [
            [
                "[Br:1][CH2:2][CH2:3][OH:4].[CH2:5]([S:7](Cl)(=[O:9])=[O:8])[CH3:6].CCOCC>C(N(CC)CC)C>[CH2:5]([S:7]([O:4][CH2:3][CH2:2][Br:1])(=[O:9])=[O:8])[CH3:6]",
                "US03930836",
                "",
                1976,
                "",
                "",
            ],
            [
                "[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([N+:12]([O-])=O)[CH:8]=[CH:7][CH:6]=2.O.[OH-].[Na+]>C(O)(=O)C.[Fe]>[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([NH2:12])[CH:8]=[CH:7][CH:6]=2 |f:2.3|",
                "US03930837",
                "",
                1976,
                "",
                "",
            ],
        ],
        columns=[
            "ReactionSmiles",
            "PatentNumber",
            "ParagraphNum",
            "Year",
            "TextMinedYield",
            "CalculatedYield",
        ],
    )

    d = tmp_path / "uspto_data"
    d.mkdir()
    in_f = d / "tmp.txt"
    out_f = d / "tmp_out.txt"
    uspto_data.to_csv(in_f, index=False, sep="\t")

    reload_uspto_data = pd.read_csv(in_f, sep="\t")
    reload_uspto_data = reload_uspto_data.replace(np.nan, "", regex=True)
    # print(uspto_data)
    # print(reload_uspto_data)
    assert (uspto_data == reload_uspto_data).all().all()

    process_uspto_dataset(input_csv=in_f, output_csv=out_f)
    out_uspto_data = pd.read_csv(out_f)
    assert out_uspto_data["rxn"].tolist()[0:2] == [
        "CCN(CC)CC.CCOCC.CCS(=O)(=O)Cl.OCCBr>>CCS(=O)(=O)OCCBr"
    ]
