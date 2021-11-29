import gc
import math
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.utils.data as data
from src.utils import create_col2idxs


class NegSampler:
    """
    Abstract class for negative sampling.
    """

    def __init__(self):
        pass

    def __call__(self, triplets: torch.Tensor):
        raise NotImplementedError


class NegDataset(data.Dataset):
    def __init__(self, triplets: torch.Tensor, neg_sampler: NegSampler):
        self.triplets = triplets
        self.neg_sampler = neg_sampler
        self.neg_triplets = neg_sampler(triplets)

    def __getitem__(self, index):
        return self.neg_triplets[index]

    def __len__(self):
        return len(self.neg_triplets)

    def regenerate(self):
        self.neg_triplets = self.neg_sampler(self.triplets)


def corrupt_triplets(triplets: torch.Tensor,
                     min_subject_id: int, max_subject_id: int,
                     min_object_id: int, max_object_id: int,
                     neg_per_pos: int, mask: Optional[torch.Tensor] = None
                     ) -> torch.Tensor:
    # examples = triplets.repeat(neg_per_pose, 1)
    # mask - which examples need to change the head, and which the tail.
    triplets = triplets.long()
    triplets = triplets[torch.randperm(triplets.size()[0])]

    corrupt_triplets = triplets.repeat(neg_per_pos, 1)

    corrupt_subjects_num = mask.long().sum().item()
    corrupt_subjects = torch.randint(
        low=min_subject_id, high=max_subject_id + 1,
        size=(corrupt_subjects_num, ), device=torch.device('cuda'))

    corrupt_objects_num = len(mask) - corrupt_subjects_num
    corrupt_objects = torch.randint(
        low=min_object_id, high=max_object_id + 1,
        size=(corrupt_objects_num, ), device=torch.device('cuda'))

    corrupt_triplets[mask, 0] = corrupt_subjects
    corrupt_triplets[~mask, 2] = corrupt_objects

    return corrupt_triplets


class HonestNegSampler(NegSampler):
    """
    Class for negative sampling. Guaranties that negative samples would not
    contain any positive samples.
    """

    def __init__(self, num_subjects: int, num_objects: int,
                 all_triplets: np.array, neg_per_pos: int):
        super().__init__()
        self.num_subjects = num_subjects
        self.num_objects = num_objects
        self.all_triplets = all_triplets
        self.neg_per_pos = neg_per_pos

        # Relation to corresponding row idxs in self.all_triplets
        self.rel2idxs = create_col2idxs(all_triplets, 1)

        self.rel2adj = {}
        for rel, idxs in self.rel2idxs.items():
            adj_values = np.ones(len(idxs))
            adj_coords = all_triplets[idxs, 0], all_triplets[idxs, 2]
            adj_shape = (self.num_objects, self.num_subjects)
            self.rel2adj[rel] = sp.csr_matrix((adj_values, adj_coords),
                                              adj_shape)

    @staticmethod
    def _sample_from_zeros(n: int, sparse: sp.csr_matrix) -> List[List[int]]:
        """
        Sample n zeros from sparse matrix.
        Parameters
        ----------
        n : int
            Number of samples to get from matrix.
        sparse : sp.csr_matrix
            Sparse matrix.
        Returns
        -------
        List[List[int]]
            List of 2-D indices of zeros.
        """
        zeros = np.argwhere(np.logical_not(sparse.todense()))
        ids = np.random.choice(range(len(zeros)), size=(n,))
        return zeros[ids].tolist()

    @staticmethod
    def _sample_by_row(num_of_iters_y: int, sparse: sp.csr_matrix,
                       part_of_zero_i: List[float], submatrix_size: int,
                       n_of_samples: int, start_idx: int,
                       end_idx: Optional[int] = None
                       ) -> list:
        """
        Sample zeros from submatrix of sparse of kind: sparse[start_idx:end_idx].
        Parameters
        ----------
        num_of_iters_y : int
        sparse : sp.csr_matrix
            Sparse matrix.
        part_of_zero_i : List[float]
            Part on n samples to get from current part of matrix.
        submatrix_size : int
            Size of submatrix (height and width).
        n_of_samples : int
            Samples to get from matrix.
        start_idx : int
            Start index of submatrix by x.
        end_idx : Optional[int]
            End index of submatrix by x.
        Returns
        -------
        list
            List of samples.
        """
        to_return = []
        for j in range(num_of_iters_y):
            to_sample = math.ceil(n_of_samples * (part_of_zero_i[j]))
            submat = sparse[start_idx:end_idx,
                     j * submatrix_size:(j + 1) * submatrix_size]
            ids_in_submat = HonestNegSampler._sample_from_zeros(to_sample,
                                                                submat)
            ids_in_mat = ids_in_submat + \
                         np.array([start_idx, j * submatrix_size])
            to_return.extend(ids_in_mat)
        j = num_of_iters_y
        if j * submatrix_size < sparse.shape[1]:
            to_sample = math.ceil(n_of_samples * (part_of_zero_i[j]))
            submat = sparse[start_idx:end_idx,
                     j * submatrix_size:]
            ids_in_submat = HonestNegSampler._sample_from_zeros(to_sample,
                                                                submat)
            ids_in_mat = ids_in_submat + \
                         np.array([start_idx, j * submatrix_size])
            to_return.extend(ids_in_mat)
        return to_return

    @staticmethod
    def _get_number_of_zeros_by_row(sparse: sp.csr_matrix,
                                    num_of_iters_y: int,
                                    submatrix_size: int,
                                    elements_in_submatrix: int,
                                    start_idx: int,
                                    end_idx: Optional[int] = None
                                    ) -> List[float]:
        """
        Get number of zeros in submatrix of sparse of kind: sparse[start_idx:end_idx].
        Parameters
        ----------
        sparse : sp.csr_matrix
            Sparse matrix.
        num_of_iters_y : int
        submatrix_size : int
            Size of submatrix (height and width).
        elements_in_submatrix : int
            Number of elements in submatrix.
        start_idx : int
            Start index of submatrix by x.
        end_idx : Optional[int]
            End index of submatrix by x.
        Returns
        -------
        List[float]
            List of number of zeros in each submatrix.
        """
        tmp = []
        for j in range(num_of_iters_y):
            tmp.append(1 - sparse[start_idx:end_idx,
                           j * submatrix_size:(
                                                      j + 1) * submatrix_size].count_nonzero()
                       / elements_in_submatrix)
        j = num_of_iters_y
        if j * submatrix_size < sparse.shape[1]:
            sub_mtr = sparse[start_idx:end_idx, j * submatrix_size:]
            tmp.append(
                1 - sub_mtr.count_nonzero() / (
                        sub_mtr.shape[0] * sub_mtr.shape[1]))
        return tmp

    @staticmethod
    def _not_exist_edges(sparse: sp.csr_matrix, n_of_samples: int,
                         submatrix_size: int = 1000
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform negative sampling.
        Parameters
        ----------
        sparse : sp.csr_matrix
            Sparse matrix.
        n_of_samples : int
            Number os samples to get.
        submatrix_size : int
            Size of submatrix (height and width).
        Returns
        -------
        np.ndarray, np.ndarray
            Negative samples (array of begin nodes and array of end nodes).
        """
        num_of_iters_x = sparse.shape[0] // submatrix_size
        num_of_iters_y = sparse.shape[1] // submatrix_size

        # count nonzero elements on each submatrix
        elements_in_submatrix = submatrix_size ** 2
        part_of_zero = []
        for i in range(num_of_iters_x):
            part_of_zero.append(
                HonestNegSampler._get_number_of_zeros_by_row(
                    sparse,
                    num_of_iters_y,
                    submatrix_size,
                    elements_in_submatrix,
                    i * submatrix_size,
                    (i + 1) * submatrix_size))
        i = num_of_iters_x
        if num_of_iters_x * submatrix_size < sparse.shape[0]:
            part_of_zero.append(
                HonestNegSampler._get_number_of_zeros_by_row(
                    sparse,
                    num_of_iters_y,
                    submatrix_size,
                    elements_in_submatrix,
                    i * submatrix_size))

        norm = sum([sum(i) for i in part_of_zero])
        part_of_zero = [[i / norm for i in lst] for lst in part_of_zero]
        result = []
        for i in range(num_of_iters_x):
            print(f"Progress: {i}/{num_of_iters_x}")
            result.extend(HonestNegSampler._sample_by_row(
                num_of_iters_y, sparse, part_of_zero[i],
                submatrix_size, n_of_samples,
                i * submatrix_size, (i + 1) * submatrix_size))
            gc.collect()
        if num_of_iters_x * submatrix_size < sparse.shape[0]:
            result.extend(
                HonestNegSampler._sample_by_row(
                    num_of_iters_y, sparse, part_of_zero[i],
                    submatrix_size, n_of_samples,
                    num_of_iters_x * submatrix_size))
        np.random.shuffle(result)
        result = np.vstack(result[:n_of_samples])
        return result[:, 0], result[:, 1]

    def negative_by_type(self, num_of_neg_samples: int, rel: int) \
            -> torch.Tensor:
        """
        Get negative samples for given relative type.
        Guaranteed no positive examples.
        Parameters
        ----------
        kg : KnowledgeGraph
        data_type : str
            Train, val or test.
        Returns
        -------
        torch.Tensor
            Negative samples.
        """

        neg_samples = HonestNegSampler._not_exist_edges(
            sparse=self.rel2adj[rel],
            n_of_samples=num_of_neg_samples)

        neg_samp = torch.zeros((num_of_neg_samples, 3))
        neg_samp[:, 0] = torch.tensor(neg_samples[0])
        neg_samp[:, 1] = torch.full((num_of_neg_samples,), rel)
        neg_samp[:, 2] = torch.tensor(neg_samples[1])
        return neg_samp

    def __call__(self, triplets: torch.Tensor):
        rel2idxs = create_col2idxs(np.array(triplets), 1)
        neg_triplets = []
        for rel, idxs in rel2idxs.items():
            neg_triplets.append(
                self.negative_by_type(self.neg_per_pos * len(idxs), rel))
        return torch.cat(neg_triplets, axis=0)


class UniformNegSampler(NegSampler):
    """
    Class for uniform negative sampling.
    Gives no guarantee that negative samples could not contain positive one.
    """

    def __init__(self, num_subjects, num_objects, neg_per_pos: int):
        super().__init__()
        self.num_subjects = num_subjects
        self.num_objects = num_objects
        self.neg_per_pos = neg_per_pos

    def __call__(self, triplets: torch.Tensor):
        # Меняем голову или хвост пары на случайный элемент
        triplets = triplets.long()
        every_corrupts_probs = torch.Tensor(
            [0.5] * len(triplets) * self.neg_per_pos)
        mask = torch.bernoulli(every_corrupts_probs).bool()
        neg_examples = corrupt_triplets(
            triplets,
            min_subject_id=0, max_subject_id=self.num_subjects - 1,
            min_object_id=0, max_object_id=self.num_objects - 1,
            neg_per_pos=self.neg_per_pos, mask=mask)
        return neg_examples


class BernoulliNegSampler(NegSampler):
    def __init__(self, train_triplets: np.array,
                 num_subjects: int, num_objects: int,
                 neg_per_pos: int):
        super().__init__()
        self.num_subjects = num_subjects
        self.num_objects = num_objects
        self.neg_per_pos = neg_per_pos
        self.head_corrupt_probs = self.get_head_corrupt_probs(train_triplets)

    def __call__(self, triplets: torch.Tensor):
        triplets = triplets.long()
        rel2idxs = create_col2idxs(np.array(triplets), 1)
        every_corrupts_probs = torch.empty(len(triplets, ))
        for rel, idxs in rel2idxs.items():
            every_corrupts_probs[idxs] = self.head_corrupt_probs[rel]
        every_corrupts_probs = every_corrupts_probs.repeat(self.neg_per_pos)
        mask = torch.bernoulli(every_corrupts_probs).bool()
        neg_examples = corrupt_triplets(
            triplets,
            min_subject_id=0, max_subject_id=self.num_subjects-1,
            min_object_id=0, max_object_id=self.num_objects-1,
            neg_per_pos=self.neg_per_pos, mask=mask)
        return neg_examples

    @staticmethod
    def get_head_corrupt_probs(triplets: np.array) -> np.array:
        rel2idxs = create_col2idxs(triplets, column=1)
        head_corrupt_probs = np.zeros((len(rel2idxs),))
        for rel, idxs in rel2idxs.items():
            rel_triplets = triplets[idxs]
            rel_head2idxs = create_col2idxs(rel_triplets, column=0)
            rel_tail2idxs = create_col2idxs(rel_triplets, column=2)
            tph = np.mean(
                [len(tails_idxs) for tails_idxs in rel_head2idxs.values()])
            hpt = np.mean(
                [len(heads_idxs) for heads_idxs in rel_tail2idxs.values()])

            head_corrupt_probs[rel] = tph / (hpt + tph)
        return head_corrupt_probs
