import logging

import pandas as pd
from rxn.utilities.logging import setup_console_logger

from rxn_negative_learning.data_generation.help_dicts_regiosqm import (
    COMPOUNDS_HALO_REACTANTS_DICT,
    HALO_REACTANT_SMILES_DICT,
    HALOGENS,
)
from rxn_negative_learning.utils.repo_utils import data_directory

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

REACTANT_RXN_CENTERS_FILE = (
    f"{data_directory() / 'source_data' / 'regiosqm' / 'compounds_smiles.csv'}"
)
OUTPUT_FILE = f"{data_directory() / 'source_data' / 'regiosqm' / 'extracted_regiosqm.csv'}"


def contains_halogen(smiles: str):
    return any([halo in smiles for halo in HALOGENS])


def load_regiosqm_data():
    """Extract the information for the reactions from the RegioSQM20 paper (10.1186/s13321-021-00490-7)"""

    df_reactant_rxn_centres = pd.read_csv(REACTANT_RXN_CENTERS_FILE, sep=r"\s+", header=None)
    df_reactant_rxn_centres.columns = ["name", "smiles", "reaction_centre"]

    halo_reactants = []
    compounds_name = []
    main_compounds = []
    reaction_centers = []
    for key, val in COMPOUNDS_HALO_REACTANTS_DICT.items():
        name = key
        matches = COMPOUNDS_HALO_REACTANTS_DICT[key]
        main_compound = (
            df_reactant_rxn_centres[df_reactant_rxn_centres.name == name].smiles.values[0].strip()
        )
        reported_positions = df_reactant_rxn_centres[
            df_reactant_rxn_centres.name == name
        ].reaction_centre.values[0]
        if reported_positions[-1] == ",":
            reported_positions = reported_positions[:-1]
        positions = [int(p.strip()) for p in reported_positions.split(",")]

        compound_halo_reactants = []
        for match in matches:
            reactants = HALO_REACTANT_SMILES_DICT[match.strip()]
            for halo_reactant in reactants:
                compound_halo_reactants.append(halo_reactant)
        if compound_halo_reactants:
            compounds_name.append(name)
            halo_reactants.append(sorted(list(set(compound_halo_reactants))))
            main_compounds.append(main_compound)
            reaction_centers.append(positions)
        else:
            logger.info(f"Incomplete data for compound {name}")

    joined_df = pd.DataFrame({
        "name": compounds_name,
        "main_reactant": main_compounds,
        "reaction_centers": reaction_centers,
        "halo_reactants": halo_reactants,
    })
    logger.info(f"Found {len(joined_df)} data points.")
    joined_df.to_csv(OUTPUT_FILE, index=False)


def main():
    setup_console_logger()
    load_regiosqm_data()


if __name__ == "__main__":
    main()
