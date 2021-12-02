from typing import Callable

import numpy as np
from pandas import DataFrame
from sklearn.metrics import pairwise_distances


def get_embedding_similarity_by_type(similarity_type: str, data: DataFrame) -> Callable[[np.array, np.array], float]:
    """
    Returns similarity function by it's name.

    Parameters
    ----------
    similarity_type : str
        Name of similarity function to return.
    data : DataFrame
        In case of using lor similarity one should provide descriptors dataframe.

    Return
    ------
    similarity_function : Callable[[np.array, np.array], float].
    -------

    """
    if similarity_type == "tanimoto":
        return tanimoto_similarity
    elif similarity_type == 'exp':
        return exp_similarity
    elif similarity_type[:3] == 'exp':
        sigma = int(similarity_type[4:])
        return lambda x, y: exp_similarity(x, y, sigma=sigma)
    elif similarity_type == "lor":
        return get_lor_similarity(data)


def tanimoto_similarity(fp1: np.array, fp2: np.array) -> float:
    """
    Return Tanimoto similarity score between two fingerprints.

    Parameters
    ----------
    fp1 : np.array
    fp2 : np.array

    Return
    ------
    similarity_score : float
        Score from [0.0, 1.0], where 1.0 is for identical sequences.
    """
    min_len = min(len(fp1), len(fp2))
    fp1 = fp1[:min_len]
    fp2 = fp2[:min_len]
    intersection = np.dot(fp1, fp2)
    return intersection / (np.sum(fp1 ** 2) + np.sum(fp2 ** 2) - intersection)


def exp_similarity(fp1, fp2, sigma=1.0):
    """
    Return exp(-||fp1 - fp2||/(sigma^2)) score between two vectors.

    Parameters
    ----------
    fp1 : np.array
    fp2 : np.array
    sigma : float
        Squared value defining dispersion.
        Default 1.0

    Return
    ------
    similarity_score : float
        Score from [0.0, 1.0], where 1.0 is for identical sequences.
    """
    squared_norm = np.linalg.norm(fp1 - fp2) ** 2
    return np.exp(-squared_norm / (sigma ** 2))


def get_lor_similarity(data: DataFrame) -> Callable[[np.array, np.array], float]:
    """
    Returns lor similarity function according to descriptors data.

    Parameters
    ----------
    data : DataFrame
        Descriptors dataframe.

    Returns
    -------
    similarity_function : Callable[[np.array, np.array], float]
    """
    def lor_distance(fp1, fp2):
        """
        Return lor similarity score between two fingerprints (similarity in hyperbolic space).

        Parameters
        ----------
        fp1 : np.array
        fp2 : np.array

        Return
        ------
        similarity_score : float
            Score from [0.0, 1.0], where 1.0 is for identical sequences.
        """
        m = fp1 * fp2
        lor_prod = m[1:].sum() - m[0]
        x = - lor_prod
        x = np.where(x < 1.0, 1.0 + 1e-6, x)
        return np.log(x + np.sqrt(x ** 2 - 1))

    dist = pairwise_distances(data, data, lor_distance)
    max_dist = dist.max()
    return lambda x, y: (max_dist - lor_distance(x, y)) / max_dist * 3


# Edges power scoring functions

def get_edges_degree_score_multiplication(edges_power):
    """
    Return function measures similarity score between 2 nodes,
    based on multiplication on their powers.

    Parameters
    ----------
    edges_power : array-like
        Array of edges powers.

    Returns
    -------
    power_score : Callable (of 2 arguments)
    """
    max_power = edges_power.max()
    min_power = edges_power.min()

    def edges_power_score_multiplication(x, y):
        x_weight = 1 - (x - min_power) / (max_power - min_power)
        y_weight = 1 - (y - min_power) / (max_power - min_power)

        return x_weight * y_weight

    return edges_power_score_multiplication
