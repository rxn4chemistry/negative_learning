import functools
import random
from enum import Enum
from math import ceil, floor

import pandas as pd
from xxhash import xxh64_intdigest


class NotSuitableParametersError(Exception):
    def __init__(self, split_ratio: float, seed: int):
        print(
            f"Sorry, could not generate appropriate splits for this values of parameters:\nsplit ratio: {split_ratio}\nseed: {seed}"
        )


class SplittingMethod(Enum):
    random = "random"
    product = "product"
    product_hash = "product_hash"
    product_tanimoto = "product_tanimoto"


class DataSplitter:
    @staticmethod
    def split(
        df: pd.DataFrame,
        reaction_column_name: str,
        idx_column_name: str,
        splitting_method: SplittingMethod,
        split_ratio: float = 0.05,
        seed: int = 0,
        validation_set: bool = False,
    ):
        """
        Function to handle the splitting of a dataset in different ways.

        Parameters
        ----------
        df: a dataframe containing at least a reaction column with reaction smiles
        reaction_column_name: the name of the column where the reactions are stored
        idx_column_name: the name of the column containing the id of the refence compound,
            this allows positives and negatives to stay in the same split
        splitting_method: the type of splitting method. Available ones are
                SplittingMethod.random = reactions are splitted randomly
                    This is NOT STABLE for a new version of the same dataset
                SplittingMethod.product = the unique products are splitted randomly:
                    reactions with the same product end up in the same split. Use this
                    instead of "product_hash" for a dataset with few products.
                    This is NOT STABLE for a new version of the same dataset.
                SplittingMethod.product_hash = the unique products are hashed and
                    then splitted according to the hash value.
                    This is STABLE for a new version of the same dataset
                SplittingMethod.product_tanimoto = not yet implemented
        split_ratio: the splitting ratio between train/test or train/test/valid
            e.g. split_ratio=0.1 means 10%test and 90%train OR 10%test and 10%valid and 80%train
                 if the validation_set flag is set to True
        seed: the seed to use to generate the dataset. Changing seed will change the splitting
        validation_set: wheter to generate also a validation set.

        """
        hash_fn = functools.partial(xxh64_intdigest, seed=seed)

        if splitting_method == SplittingMethod.random:
            random.seed(seed)
            # The hash is the idx of the reaction
            df["hash"] = df[idx_column_name]
            list_of_unique_ids = sorted([key for key, value in df["hash"].value_counts().items()])
            number_of_unique_ids = len(list_of_unique_ids)

            list_of_test_ids = random.sample(
                list_of_unique_ids, k=floor(number_of_unique_ids * split_ratio)
            )

            list_of_unique_ids = sorted(list(set(list_of_unique_ids) - set(list_of_test_ids)))

            if validation_set:
                list_of_valid_ids = random.sample(
                    list_of_unique_ids, k=ceil(number_of_unique_ids * split_ratio)
                )
                list_of_unique_ids = list(set(list_of_unique_ids) - set(list_of_valid_ids))

            df[f"sratio{split_ratio}_{splitting_method.value}_seed{seed}"] = df.hash.apply(
                lambda x: "test"
                if x in list_of_test_ids
                else "train"
                if x in list_of_unique_ids
                else "valid"
            )
            print(f"Stats for seed {seed} with method '{splitting_method.value}'.")
            print(
                df[f"sratio{split_ratio}_{splitting_method.value}_seed{seed}"].value_counts(
                    normalize=True
                )
            )

        elif splitting_method == SplittingMethod.product_hash:
            df["hash"] = (
                df[reaction_column_name]
                .apply(lambda value: value.split(">>")[1])
                .apply(lambda value: hash_fn(value))
            )
            if not validation_set:
                df[f"sratio{split_ratio}_{splitting_method.value}_seed{seed}"] = df.hash.apply(
                    lambda x: "test" if x < split_ratio * 2**64 else "train"
                )
            else:
                df[f"sratio{split_ratio}_{splitting_method.value}_seed{seed}"] = df.hash.apply(
                    lambda x: "test"
                    if x < split_ratio * 2**64
                    else "train"
                    if x >= split_ratio * 2 * 2**64
                    else "valid"
                )
                if (
                    len(
                        df.loc[
                            df[f"sratio{split_ratio}_{splitting_method.value}_seed{seed}"]
                            == "valid"
                        ]
                    )
                    == 0
                    or len(
                        df.loc[
                            df[f"sratio{split_ratio}_{splitting_method.value}_seed{seed}"] == "test"
                        ]
                    )
                    == 0
                ):
                    raise NotSuitableParametersError(split_ratio, seed)

        elif splitting_method == SplittingMethod.product:
            random.seed(seed)
            # The hash is the product smiles
            df["hash"] = df[reaction_column_name].apply(lambda value: value.split(">>")[1])
            list_of_unique_products = sorted([
                key for key, value in df["hash"].value_counts().items()
            ])
            number_of_unique_products = len(list_of_unique_products)

            random.seed(seed)
            list_of_test_products = random.sample(
                list_of_unique_products, k=ceil(number_of_unique_products * split_ratio)
            )

            list_of_unique_products = sorted(
                list(set(list_of_unique_products) - set(list_of_test_products))
            )

            if validation_set:
                list_of_valid_products = random.sample(
                    list_of_unique_products,
                    k=ceil(number_of_unique_products * split_ratio),
                )
                list_of_unique_products = list(
                    set(list_of_unique_products) - set(list_of_valid_products)
                )

            df[f"sratio{split_ratio}_{splitting_method.value}_seed{seed}"] = df.hash.apply(
                lambda x: "test"
                if x in list_of_test_products
                else "train"
                if x in list_of_unique_products
                else "valid"
            )

        elif splitting_method == SplittingMethod.product_tanimoto:
            raise NotImplementedError(
                f"The method '{splitting_method.value}' has not been implemented yet."
            )
