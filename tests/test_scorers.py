import pytest
from rxn_negative_learning.models.scorers.ideal_scorer import IdealScorer
from rxn_negative_learning.models.scorers.levenstein_ideal_scorer import LevenshteinIdealScorer
from rxn_negative_learning.models.scorers.svm_scorer import SVMScorer
from rxn_negative_learning.models.scorers.tanimoto_ideal_scorer import TanimotoIdealScorer
from rxn_negative_learning.utils.repo_utils import models_directory


def test_ideal_scorer():
    pos_rxns = ["A.A>>B", "C.C>>D"]
    neg_rxns = ["A.A>>C", "C.C>>B"]
    ideal_scorer = IdealScorer(pos_rxns, neg_rxns)

    result = ideal_scorer(["A.A>>B", "C.C>>D", "A.A>>C", "C.C>>B"])
    assert result == [1, 1, 0, 0]

    # With shift
    ideal_scorer = IdealScorer(pos_rxns, neg_rxns, shift=0.5)

    result = ideal_scorer(["A.A>>B", "C.C>>D", "A.A>>C", "C.C>>B"])
    assert result == [0.5, 0.5, -0.5, -0.5]


def test_levenshtein_ideal_scorer():
    pos_rxns = ["ola>>ciao"]
    levenshtein_scorer = LevenshteinIdealScorer(pos_rxns)

    result = levenshtein_scorer(
        ["ola>>ciao", "ola>>miao", "cola>>vola", "ola>>"],
    )
    assert result == [1, pytest.approx(0.5), 0, 0]

    # With shift
    levenshtein_scorer = LevenshteinIdealScorer(pos_rxns, shift=0.5)

    result = levenshtein_scorer(
        ["ola>>ciao", "ola>>miao", "cola>>vola", "ola>>"],
    )
    assert result == [0.5, pytest.approx(0.5 - 0.5), -0.5, -0.5]

    # With negative reactions grounding
    neg_rxns = ["ola>>miao", "ola>>wow"]
    levenshtein_scorer = LevenshteinIdealScorer(pos_rxns, neg_rxns)

    result = levenshtein_scorer(
        ["ola>>ciao", "ola>>miao", "cola>>vola", "ola>>wow", "ola>>"],
    )
    assert result == [1, 0, 0, 0, 0]


def test_tanimoto_ideal_scorer():
    pos_rxns = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1"]
    tanimoto_scorer = TanimotoIdealScorer(pos_rxns)

    result = tanimoto_scorer([
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>",
        "CO.CCCC>>[Na+]",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1ccn[nH]1",
    ])
    assert result == [1, 0, 0, pytest.approx(0.40540540540540543)]

    # With shift
    tanimoto_scorer = TanimotoIdealScorer(pos_rxns, shift=0.5)

    result = tanimoto_scorer([
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>",
        "CO.CCCC>>[Na+]",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1ccn[nH]1",
    ])
    assert result == [0.5, -0.5, -0.5, pytest.approx(0.40540540540540543 - 0.5)]

    # With negative grounding
    neg_rxns = ["O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1ccn[nH]1"]
    tanimoto_scorer = TanimotoIdealScorer(pos_rxns, neg_rxns)

    result = tanimoto_scorer([
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1cn[nH]c1",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>",
        "CO.CCCC>>[Na+]",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>Brc1ccn[nH]1",
    ])
    assert result == [1, 0, 0, 0]


def test_svm_scorer():
    model_path = models_directory() / "old" / "svm_scorer"
    print(model_path)
    svm_scorer = SVMScorer(model_path)

    result = svm_scorer([
        "O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1[nH]cc(-c2ccncc2)cc1Br",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>",
        "CO.CCCC>>[Na+]",
        "O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1ccc(-c2ccncc2Br)c[nH]1",
    ])
    assert result == [1, 0, 0, 1]

    # With shift
    svm_scorer = SVMScorer(model_path, shift=0.5)

    result = svm_scorer([
        "O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1[nH]cc(-c2ccncc2)cc1Br",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>",
        "CO.CCCC>>[Na+]",
        "O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1ccc(-c2ccncc2Br)c[nH]1",
    ])
    assert result == [0.5, -0.5, -0.5, 0.5]

    # With grounding
    pos_rxns = ["O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1[nH]cc(-c2ccncc2)cc1Br"]
    neg_rxns = ["O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1ccc(-c2ccncc2Br)c[nH]1"]
    svm_scorer = SVMScorer(
        model_path=model_path, positive_reactions=pos_rxns, negative_reactions=neg_rxns
    )

    result = svm_scorer([
        "O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1[nH]cc(-c2ccncc2)cc1Br",
        "O=C1CCC(=O)N1Br.c1cn[nH]c1>>",
        "CO.CCCC>>[Na+]",
        "O=C1CCC(=O)N1Br.O=c1ccc(-c2ccncc2)c[nH]1>>O=c1ccc(-c2ccncc2Br)c[nH]1",
    ])

    assert result == [1, 0, 0, 0]
