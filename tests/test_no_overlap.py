import pandas as pd
from rxn_negative_learning.scripts.remove_overlap import remove_overlap


def test_no_overlap(tmp_path):
    data_ref = pd.DataFrame(
        [
            "CCC.[Na+].O>>CCC",
            "CCC.[Na+].O>>[Na+]",
            "CCC.CCC.O>>[Na+]",
            "CCC.CCC.O>>O.O",
            "CCC.CCC.O>>[Cl-]",
        ],
        columns=["rxn"],
    )
    d = tmp_path / "data"
    d.mkdir()
    in_f = d / "tmp.txt"
    data_ref.to_csv(in_f, index=False)

    data_1 = pd.DataFrame(["O.O>>O.O"], columns=["rxn"])
    data_2 = pd.DataFrame(
        [
            "CCC.[Na+].O>>CCC",
            "CCC.[Na+].O>>[Na+]",
        ],
        columns=["rxn"],
    )
    in_f1 = d / "tmp1.txt"
    in_f2 = d / "tmp2.txt"
    data_1.to_csv(d / in_f1, index=False)
    data_2.to_csv(d / in_f2, index=False)

    remove_overlap(in_f, [in_f1, in_f2], d / "out.txt", "rxn")
    df = pd.read_csv(d / "out.txt")
    assert df["rxn"].tolist() == ["CCC.CCC.O>>[Cl-]"]
