"""
Definition of the different metrics.
"""

import math
from typing import Dict, List, Sequence, Tuple, TypeVar

from rxn.utilities.containers import chunker

T = TypeVar("T")


def top_n_accuracy_neg(neg_ground_truth: Sequence[T], predictions: Sequence[T]) -> Dict[int, float]:
    """
    Compute the top-n accuracy values given a sequence of negative yargets per prediction.
    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_sequence_multiplier().
    Returns:
        Dictionary of top-n negative accuracy values.
    """
    multiplier = get_sequence_multiplier(ground_truth=neg_ground_truth, predictions=predictions)
    # we will count, for each "n", how many predictions are correct
    correct_for_topn: List[int] = [0 for _ in range(multiplier)]
    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    for ngts, predictions in zip(neg_ground_truth, prediction_chunks):
        for i in range(multiplier):
            correct = any([ngt in predictions[: i + 1] for ngt in ngts])
            correct_for_topn[i] += int(correct)

    return {i + 1: correct_for_topn[i] / len(neg_ground_truth) for i in range(multiplier)}


def top_n_accuracy(ground_truth: Sequence[T], predictions: Sequence[T]) -> Dict[int, float]:
    """
    Compute the top-n accuracy values.
    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_sequence_multiplier().
    Returns:
        Dictionary of top-n accuracy values.
    """
    multiplier = get_sequence_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will count, for each "n", how many predictions are correct
    correct_for_topn: List[int] = [0 for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    for gt, predictions in zip(ground_truth, prediction_chunks):
        for i in range(multiplier):
            correct = gt in predictions[: i + 1]
            correct_for_topn[i] += int(correct)

    return {i + 1: correct_for_topn[i] / len(ground_truth) for i in range(multiplier)}


def top_n_invalids(ground_truth: Sequence[T], predictions: Sequence[T]) -> Dict[int, float]:
    """
    Compute the top-n invalid percentage values.
    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_sequence_multiplier().
    Returns:
        Dictionary of top-n percentages values.
    """
    multiplier = get_sequence_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will count, for each "n", how many predictions are invalid
    invalids_for_topn: List[int] = [0 for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    for _, predictions in zip(ground_truth, prediction_chunks):
        for i in range(multiplier):
            invalid = predictions[i] == ""
            invalids_for_topn[i] += int(invalid)

    return {i + 1: invalids_for_topn[i] / len(ground_truth) for i in range(multiplier)}


def get_sequence_multiplier(ground_truth: Sequence[T], predictions: Sequence[T]) -> int:
    """
    Get the multiplier for the number of predictions by ground truth sample.
    Raises:
        ValueError: if the lists have inadequate sizes (possibly forwarded
            from get_multiplier).
    """
    n_gt = len(ground_truth)
    n_pred = len(predictions)

    return get_multiplier(n_gt, n_pred)


def get_multipliers(a: int, b: int) -> Tuple[int, int]:
    """
    Get the multipliers m_a and m_b so that m_a * a == m_b * b.
    Raises:
        ValueError: If one of the numbers is not strictly positive.
    Returns:
        Tuple: multiplier for a, multiplier for b.
    """
    if a < 1 or b < 1:
        raise ValueError(
            f"Can't determine multipliers for non-strictly-positive numbers ({a} and {b})"
        )

    # Lowest common multiplier, https://stackoverflow.com/a/51716959
    lcm = abs(a * b) // math.gcd(a, b)

    return lcm // a, lcm // b


def get_multiplier(a: int, b: int) -> int:
    """
    Get the multiplier m so that m * a == b.
    Raises:
        ValueError: If b is not exactly a multiplier of a (possibly forwarded from
            get_multipliers).
    Returns:
        multiplier for a.
    """

    m_a, m_b = get_multipliers(a, b)

    # if the multiplier for b is not 1, it means that b is not a multiple of a.
    if m_b != 1:
        raise ValueError(f"{b} is not a multiple of {a}.")

    return m_a
