import torch
from torch.nn import Module
from src.chemical_embeddings import EmbeddingsInitializer
from typing import Optional


class TriVec(Module):
    """
    TriVec model.
    Knowledge graph link prediction.

    Parameters
    ----------
    dim : int
        Embedding size.
    ent_total : int
        Num of different entities.
    rel_total : int
        Num of different relatives.


    Attributes
    ----------
    dim : int
        Embedding size.
    ent_total : int
        Num of different entities.
    rel_total : int
        Num of different relatives.
    ent_i : nn.Embedding
        i-th embedding for entities ( i in [1:3] )
        embed_head = embed_tail.
    rel_i : nn.Embedding
        i-th embedding ( i in [1:3] )

    Notes
    -----
    Entities embedding: (e_1, e_2, e_3). No difference between embeddings for
    head node and tail node.
    Relative embedding: (rel_1, rel_2, rel_3).
    Score Function: negative softplus loss
    (e.i. loss(score) = softplus(-y*score), where y = 1 for positive triples
    and -1 for negative) with L3 regularization.

    More info here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7233093/
    """

    def __init__(self, ent_total: int, rel_total: int,
                 embedding_init: EmbeddingsInitializer, dim: int = 100):
        super(TriVec, self).__init__()
        self.dim = dim
        self.ent_total = ent_total
        self.rel_total = rel_total
        self.regularization_const = 0.03
        self.ent_1 = embedding_init.init_entities_embeddings(self.ent_total, self.dim)
        self.ent_2 = embedding_init.init_entities_embeddings(self.ent_total, self.dim)
        self.ent_3 = embedding_init.init_entities_embeddings(self.ent_total, self.dim)
        self.rel_1 = embedding_init.init_rel_embeddings(self.rel_total, self.dim)
        self.rel_2 = embedding_init.init_rel_embeddings(self.rel_total, self.dim)
        self.rel_3 = embedding_init.init_rel_embeddings(self.rel_total, self.dim)

    def _get_emb(self, h_idx, t_idx, r_idx):
        h_1 = self.ent_1(h_idx)
        h_2 = self.ent_2(h_idx)
        h_3 = self.ent_3(h_idx)
        t_1 = self.ent_1(t_idx)
        t_2 = self.ent_2(t_idx)
        t_3 = self.ent_3(t_idx)
        r_1 = self.rel_1(r_idx)
        r_2 = self.rel_2(r_idx)
        r_3 = self.rel_3(r_idx)

        return h_1, h_2, h_3, t_1, t_2, t_3, r_1, r_2, r_3

    def get_score(self, h_idx: torch.tensor, t_idx: torch.tensor, r_idx: torch.tensor) -> torch.tensor:
        """
        For all triples return values of scoring function for each triple.

        Parameters
        ----------
        h_idx : torch.tensor
            Indices of head-nodes of triples.
        t_idx : torch.tensor
            Indices of tail-nodes of triples.
        r_idx : torch.tensor
            Indices of relation-types of triples.

        Returns
        -------
        tensor
            Values of scoring function for each triple.
        """
        h_1, h_2, h_3, t_1, t_2, t_3, r_1, r_2, r_3 = self._get_emb(h_idx,
                                                                    t_idx,
                                                                    r_idx)

        return (h_1 * r_1 * t_3).sum(dim=1) + (h_2 * r_2 * t_2).sum(dim=1) + (
                h_3 * r_3 * t_1).sum(dim=1)

    @staticmethod
    def _get_indexes_from_data(data):
        h_idx = data[:, 0]
        t_idx = data[:, 2]
        r_idx = data[:, 1]
        return h_idx, t_idx, r_idx

    def forward(self, data):
        h_idx, t_idx, r_idx = self._get_indexes_from_data(data)
        score = self.get_score(h_idx, t_idx, r_idx)
        return score

    def regularization(self, data):
        h_idx, t_idx, r_idx = self._get_indexes_from_data(data)
        h_1, h_2, h_3, t_1, t_2, t_3, r_1, r_2, r_3 = self._get_emb(h_idx,
                                                                    t_idx,
                                                                    r_idx)
        return self.regularization_const * (torch.mean(torch.abs(h_1) ** 3, dim=1) +
                 torch.mean(torch.abs(h_2) ** 3, dim=1) +
                 torch.mean(torch.abs(h_3) ** 3, dim=1) +
                 torch.mean(torch.abs(t_1) ** 3, dim=1) +
                 torch.mean(torch.abs(t_2) ** 3, dim=1) +
                 torch.mean(torch.abs(t_3) ** 3, dim=1) +
                 torch.mean(torch.abs(r_1) ** 3, dim=1) +
                 torch.mean(torch.abs(r_2) ** 3, dim=1) +
                 torch.mean(torch.abs(r_3) ** 3, dim=1)) / 3

    def predict(self, data):
        score = self.forward(data)
        return score.cpu().data.numpy()


class TriVecWeighted(TriVec):
    def __init__(self, ent_total: int, rel_total: int,
                 embedding_init: EmbeddingsInitializer, weights_similarity,
                 weights_edges_degree: Optional, dim=100):
        super().__init__(ent_total, rel_total, embedding_init, dim)
        self.weights_similarity = weights_similarity
        self.weights_edges_power = weights_edges_degree

    def _get_node_embeddings_and_weights(self, pairs):
        h_idx, t_idx = pairs[:, 0], pairs[:, 1]
        h_1, h_2, h_3, t_1, t_2, t_3, _, _, _ = \
            self._get_emb(h_idx, t_idx, torch.empty(0,
                                                    dtype=torch.long,
                                                    device=h_idx.device))
        weights_similarity = self.weights_similarity[h_idx, t_idx]
        return h_1, h_2, h_3, t_1, t_2, t_3, weights_similarity

    def get_weighted_score(self, pairs):
        h_idx, t_idx = pairs[:, 0], pairs[:, 1]
        h_1, h_2, h_3, t_1, t_2, t_3, _, _, _ = \
            self._get_emb(h_idx, t_idx, torch.empty(0,
                                                    dtype=torch.long,
                                                    device=h_idx.device))
        weights_similarity = self.weights_similarity[h_idx, t_idx]
        if self.weights_edges_power is None:
            return ((h_1 * t_3).sum(dim=1) + (h_2 * t_2).sum(dim=1) + (
                    h_3 * t_1).sum(dim=1)) * weights_similarity

        weights_edges_power = self.weights_edges_power[h_idx, t_idx]

        return ((h_1 * t_3).sum(dim=1) + (h_2 * t_2).sum(dim=1) + (
                h_3 * t_1).sum(dim=1)) * weights_similarity * weights_edges_power

    def get_embed_and_weights(self, pairs):
        h_idx, t_idx = pairs[:, 0], pairs[:, 1]
        h_1, h_2, h_3, t_1, t_2, t_3, _, _, _ = \
            self._get_emb(h_idx, t_idx, torch.empty(0,
                                                    dtype=torch.long,
                                                    device=h_idx.device))
        weights_similarity = self.weights_similarity[h_idx, t_idx]

        h_emb = torch.cat([h_1, h_2, h_3], dim=0)
        t_emb = torch.cat([t_1, t_2, t_3], dim=0)

        if self.weights_edges_power is None:
            weights = torch.cat([weights_similarity, weights_similarity, weights_similarity]).reshape(-1, 1)
            return h_emb, t_emb, weights

        weights_edges_power = self.weights_edges_power[h_idx, t_idx]
        weights = torch.cat([weights_similarity * weights_edges_power,
                             weights_similarity * weights_edges_power,
                             weights_similarity * weights_edges_power]).reshape(-1, 1)

        return h_emb, t_emb, weights


class SPTriVecWeighted(TriVecWeighted):
    """
    Model class with StayPositive negative sampling.

    Parameters
    ----------
    I : int
        Negative sampling parameter. Value of score function will be in range [-I, I].

    Attributes
    ----------
    I : int
        Negative sampling parameter. Value of score function will be in range [-I, I].

    Notes
    -----
    For more details about Stay Positive see: https://arxiv.org/pdf/1812.06410.pdf
    """
    def __init__(self, ent_total: int, rel_total: int,
                 embedding_init: EmbeddingsInitializer, weights_similarity,
                 weights_edges_degree: Optional, dim=100, I=5):
        super().__init__(ent_total, rel_total, embedding_init,
                         weights_similarity, weights_edges_degree, dim)
        self.I = I

    def _get_emb(self, h_idx, t_idx, r_idx):
        embeddings = super()._get_emb(h_idx, t_idx, r_idx)
        return tuple(torch.tanh(emb) for emb in embeddings)

    def get_weighted_score(self, pairs):
        return super().get_weighted_score(pairs) * self.I / (
                           3 * self.dim)

    def reg_score(self, data):
        h_idx, t_idx, r_idx = self._get_indexes_from_data(data)
        # h_idx - все возможные головы в батче (без повторений)
        # t_idx - все возможные хвосты в батче (без повторений)
        # r_idx - все возможные отношения в батче (без повторений)
        # Считается скор по всем тройкам, образованным h_idx, t_idx, r_idx
        # Sum_h_idx Sum_r_idx Sum_t_idx Score(h, r, t)
        h_idx, t_idx, r_idx = torch.unique(h_idx), torch.unique(t_idx), torch.unique(r_idx)
        h_1, h_2, h_3, t_1, t_2, t_3, r_1, r_2, r_3 = self._get_emb(h_idx,
                                                                    t_idx,
                                                                    r_idx)
        first_sum_part = (h_1.sum(0) * r_1.sum(0) * t_3.sum(0)).sum()
        second_sum_part = (h_2.sum(0) * r_2.sum(0) * t_2.sum(0)).sum()
        third_sum_part = (h_3.sum(0) * r_3.sum(0) * t_1.sum(0)).sum()
        return (first_sum_part + second_sum_part + third_sum_part) * self.I / (
                3 * self.dim)