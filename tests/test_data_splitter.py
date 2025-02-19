# import pytest
# import pandas as pd
# from rxn_negative_learning.data_splitter import SplittingMethod, DataSplitter
# from rxn_negative_learning.data_splitter import NotSuitableParametersError
#
# @pytest.fixture
# def dataframe():
#     df = pd.DataFrame({
#         "rxn": [
#             "c1(ccn(n1)C)C(F)F.O=C1N(Br)C(=O)CC1>>Cn1nc(C(F)F)cc1Br",
#             "c1c(ccc(c1)CSc1ccnc(c1)N)OC.BrBr>>COc1nccc2occ(-c3ccccc3Br)c12",
#             "c1nc(c2c(c1)occ2c1ccccc1)OC.BrBr>>COc1nccc2occ(-c3ccccc3Br)c12",
#             "n1cccn1c1ncccn1.BrBr>>Brc1ccnn1-c1ncccn1",
#             "c1ccc2c(c1)nc(n2S(=O)(=O)C1CCCC1)N.C1CC(=O)N(C1=O)I>>Nc1cc(I)ccn1",
#             "c1ccc(cn1)O.[Na]I.[Na+].[O-]Cl>>Oc1cncc(Cl)c1",
#             "c1(nc(c2c(n1)occ2c1ccccc1)C(F)(F)F)SC.BrBr>>Nc1cc(I)ccn1",
#             "c1cccc(n1)N.II>>Nc1cc(I)ccn1",
#             "c1c(ncc(n1)N)C(=O)OCC.O=C1N(Br)C(=O)CC1>>CCOC(=O)c1ncc(N)nc1Br",
#             "c1ccc2c(n1)ccn2c1ccc(cc1)OC.O=C1N(Br)C(=O)CC1>>Cn1nc(C(F)F)cc1Br"
#         ]
#     })
#     return df
#
# def test_data_splitter_random(dataframe):
#     old_dataframe = dataframe.copy()
#     data_splitter = DataSplitter.split(
#         df = dataframe,
#         reaction_column_name='rxn',
#         splitting_method=SplittingMethod.random,
#         split_ratio=0.1,
#         seed=22,
#         validation_set=False)
#     assert old_dataframe.rxn.to_list() == dataframe.rxn.to_list()
#     assert dataframe.random_seed22.to_list() == [
#         "train",
#         "train",
#         "test",
#         "train",
#         "train",
#         "train",
#         "train",
#         "train",
#         "train",
#         "train"
#     ]
#
#     data_splitter = DataSplitter.split(
#         df = dataframe,
#         reaction_column_name='rxn',
#         splitting_method=SplittingMethod.random,
#         split_ratio=0.1,
#         seed=22,
#         validation_set=True)
#
#     assert old_dataframe.rxn.to_list() == dataframe.rxn.to_list()
#     assert dataframe.random_seed22.to_list() == [
#         "train",
#         "train",
#         "test",
#         "train",
#         "train",
#         "train",
#         "train",
#         "train",
#         "train",
#         "valid"
#     ]
#
# def test_data_splitter_product(dataframe):
#     old_dataframe = dataframe.copy()
#     data_splitter = DataSplitter.split(
#             df=dataframe,
#             reaction_column_name='rxn',
#             splitting_method=SplittingMethod.product,
#             split_ratio=0.1,
#             seed=22,
#             validation_set=False)
#     assert old_dataframe.rxn.to_list() == dataframe.rxn.to_list()
#     assert dataframe.product_seed22.to_list() == [
#             "train",
#             "train",
#             "train",
#             "train",
#             "train",
#             "train",
#             "train",
#             "train",
#             "test",
#             "train"
#         ]
#     assert dataframe.product_seed22.to_list()[1:3] == ['train','train']
#     assert dataframe.product_seed22.to_list()[6:8] == ['train','train']
#
#     data_splitter = DataSplitter.split(
#             df=dataframe,
#             reaction_column_name='rxn',
#             splitting_method=SplittingMethod.product,
#             split_ratio=0.1,
#             seed=22,
#             validation_set=True)
#
#     assert old_dataframe.rxn.to_list() == dataframe.rxn.to_list()
#     assert dataframe.product_seed22.to_list() == [
#             "train",
#             "valid",
#             "valid",
#             "train",
#             "train",
#             "train",
#             "train",
#             "train",
#             "test",
#             "train",
#     ]
#
# def test_data_splitter_product_hash(dataframe):
#     old_dataframe = dataframe.copy()
#     data_splitter = DataSplitter.split(
#             df=dataframe,
#             reaction_column_name='rxn',
#             splitting_method=SplittingMethod.product_hash,
#             split_ratio=0.1,
#             seed=22,
#             validation_set=False)
#     assert old_dataframe.rxn.to_list() == dataframe.rxn.to_list()
#     assert dataframe.product_hash_seed22.to_list() == [
#             "train",
#             "train",
#             "train",
#             "train",
#             "train",
#             "test",
#             "train",
#             "train",
#             "train",
#             "train",
#         ]
#
#     assert dataframe.product_hash_seed22.to_list()[1:3] == ['train','train']
#     assert dataframe.product_hash_seed22.to_list()[6:8] == ['train','train']
#
#     data_splitter = DataSplitter.split(
#             df=dataframe,
#             reaction_column_name='rxn',
#             splitting_method=SplittingMethod.product_hash,
#             split_ratio=0.1,
#             seed=22,
#             validation_set=True)
#
#     assert old_dataframe.rxn.to_list() == dataframe.rxn.to_list()
#     assert dataframe.product_hash_seed22.to_list() == [
#             "train",
#             "train",
#             "train",
#             "train",
#             "valid",
#             "test",
#             "valid",
#             "valid",
#             "train",
#             "train"
#     ]
#     assert dataframe.product_hash_seed22.to_list()[1:3] == ['train','train']
#     assert dataframe.product_hash_seed22.to_list()[4] == 'valid'
#     assert dataframe.product_hash_seed22.to_list()[6:8] == ['valid','valid']
#
# def test_data_splitter_product_hash_not_suitable_params(dataframe):
#     old_dataframe = dataframe.copy()
#     with pytest.raises(NotSuitableParametersError):
#         data_splitter = DataSplitter.split(
#             df=dataframe,
#             reaction_column_name='rxn',
#             splitting_method=SplittingMethod.product_hash,
#             split_ratio=0.08,
#             seed=122,
#             validation_set=True)
#
# def test_data_splitter_product_tanimoto(dataframe):
#     old_dataframe = dataframe.copy()
#     with pytest.raises(NotImplementedError):
#         data_splitter = DataSplitter.split(
#             df=dataframe,
#             reaction_column_name='rxn',
#             splitting_method=SplittingMethod.product_tanimoto,
#             split_ratio=0.1,
#             seed=22,
#             validation_set=False)
