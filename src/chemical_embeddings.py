import pandas as pd
from torch import nn
import torch


class EmbeddingsInitializer:
    """
    Base class for embedding initializers.

    """
    def __init__(self):
        pass
    
    def _default_init(self, dim1: int, dim2: int) -> nn.Embedding:
        """
        Default init for embedding (xavier init)
        
        Parameters
        ----------
        dim1 : int
        dim2 : int

        Returns
        -------
        embeddings : nn.Embeddings
            Initialized embeddings.
        """
        embed = nn.Embedding(dim1, dim2)
        nn.init.xavier_uniform_(embed.weight.data)
        return embed

    def init_entities_embeddings(self, dim1: int, dim2: int) -> nn.Embedding:
        """
        Return initialized embeddings for entities.

        Parameters
        ----------
        dim1 : int
            Dimension 0 size
        dim2 : int
            Dimension 1 size
        return self._default_init(dim1, dim2)
        """
        return self._default_init(dim1, dim2)

    def init_rel_embeddings(self, dim1: int, dim2: int) -> nn.Embedding:
        """
        Return initialized embeddings for relations.

        Parameters
        ----------
        dim1 : int
            Dimension 0 size
        dim2 : int
            Dimension 1 size
        return self._default_init(dim1, dim2)
        """
        return self._default_init(dim1, dim2)

    
class VectorEmbeddingsInitializer(EmbeddingsInitializer):
    """
    Initialize entities embedding with some vectors.
    Relation embeddings have default init.

    Parameters
    ----------
    path_to_embedding : str
        Path to csv file with embeddings.
    """
    def __init__(self, path_to_embedding: str):
        super().__init__()
        self.embeddings = pd.read_csv(path_to_embedding, index_col=0).values

    def init_entities_embeddings(self, dim1: int, dim2: int) -> nn.Embedding:
        """
        Return initialized embeddings for entities.

        Parameters
        ----------
        dim1 : int
            Dimension 0 size
        dim2 : int
            Dimension 1 size

        Return
        ------
        embeddings : nn.Embedding
            Initialized embeddings
        
        Raises
        ------
        ValueError:
            If dim2 != self.embeddings.shape[1]
        """
        if self.embeddings.shape[1] != dim2:
            raise ValueError(f"VectorEmbeddings dim2 fixed. "
                             f"Must be equal {self.embeddings.shape[1]}")
        weight = torch.tensor(self.embeddings, dtype=torch.float)

        # if dim1 > num_of_drugs then embeddings [num_of_drugs, dim1-1] will be
        # initialized with xavier uniform distribution
        if dim1 > self.embeddings.shape[0]:
            w = torch.empty((dim1 - self.embeddings.shape[0], dim2),
                            dtype=torch.float)
            nn.init.xavier_uniform_(w)
            weight = torch.cat((weight, w), axis=0)

        return nn.Embedding.from_pretrained(weight)
