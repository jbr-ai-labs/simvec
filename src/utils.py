import numpy as np
import torch.nn as nn


def create_col2idxs(triplets: np.array, column: int):
    # used code from answer for
    # https://stackoverflow.com/questions/43442309/numpy-create-a-dict-by-grouping-values-in-a-column-by-another-column-values
    # Return dict from column value to corresponding edges indexes in triplets
    order = triplets[:, column].argsort()  # sort by rel if not already sorted
    triplets_sorted = triplets[order]
    s0 = np.concatenate(([0], np.flatnonzero(
        triplets_sorted[1:, column] > triplets_sorted[:-1, column]) + 1,
                         [triplets_sorted.shape[0]]))

    idxs = triplets_sorted[s0[:-1], column]
    return {idxs[i]: order[s0[i]:s0[i + 1]] for i in
            range(len(s0) - 1)}  # Dict[int, np.array]


def switch_grad_mode(model: nn.Module , requires_grad: bool=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def switch_model_mode(model: nn.Module, train: bool=True):
    if train:
        model.train()
    else:
        model.eval()
