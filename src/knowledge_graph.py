from typing import List, Tuple, Optional
from omegaconf import DictConfig
from abc import abstractmethod
from pandas import concat, DataFrame, read_csv, get_dummies
from constants import KG_CONST, PARZEN_WINDOW_PARAMS
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import pickle
import os

from src.node_similarity_estimator import exp_similarity, get_embedding_similarity_by_type


class KnowledgeGraph:
    """
    Class representing Knowledge graph.
    Preprocess and store dataframes for model evaluation.

    Parameters
    ----------
    data_path : str
        Path to the data folder relative to the working directory.
    use_proteins : bool
        Whether to use protein nodes or not.
    use_proteins_on_validation : bool
        Whether to use protein nodes on validation.
    Attributes
    ----------
    data_path : str
        Path to the data folder relative to the working directory.
    use_proteins : bool
        Whether to use protein nodes or not.
    use_prot_on_val : bool
        Whether to use protein nodes on validation.
    use_reversed_edges : bool
        Whether to add edge reverse duplicate or not (for drug-drug relations).
    ent_maps : DataFrame
        Mapping between STITCH ID of node, int id in data.
        WARNING: contains only drugs!
    rel_maps : DataFrame
        Mapping between CID ID of side effect, int id and name.
    _df_train : DataFrame
        DataFrame with training triples.
    _df_val : DataFrame
        DataFrame with validate triples.
    _df_test : DataFrame
        DataFrame with test triples.
    _df_drug : DataFrame
        DataFrame with all (train, test, validate) triples.
    size_train : int
        Number of training triples.
    size_val : int
        Number of validate triples.
    size_test : int
        Number of test triples.
    _list_of_se_triples : List[DataFrame]
        For each se list contains dataframe with all triples for this se.
    _counts_per_fold : DataFrame
        For each se dataframe contains num of triples in test data.
    """
    def __init__(self, cfg: DictConfig, data_path: str = ".",
                 use_proteins: bool = False,
                 use_proteins_on_validation: bool = False,
                 use_reversed_edges: bool = False):
        if use_proteins_on_validation and not use_proteins:
            raise ValueError("Can use proteins on validation only "
                             "if use_proteins = True")
        self.data_path = data_path
        self.use_proteins = use_proteins
        self.use_prot_on_val = use_proteins_on_validation
        self.use_reversed_edges = use_reversed_edges
        self.ent_maps = None
        self.rel_maps = None
        self._df_train, self._df_val, self._df_test, self._df_drug \
            = self._load_polypharmacy_data(cfg)
        self.size_train = len(self._df_train)
        self.size_val = len(self._df_val)
        self.size_test = len(self._df_test)

    def _load_polypharmacy_data(self, cfg) -> Tuple[DataFrame, DataFrame, DataFrame,
                                               DataFrame]:
        """
        Load polypharmacy data.

        Returns
        -------
        Tuple[DataFrame, DataFrame, DataFrame, np.array] :
            Train, validate, test dataframes and array with all triplec.
        """
        if self.use_reversed_edges:
            # get train data only with drugs
            df_train = self._get_df_with_reversed_edges(
                read_csv(self.data_path + cfg.data_const.drug_train))

            # get val data only with drugs
            df_val = self._get_df_with_reversed_edges(
                read_csv(self.data_path + cfg.data_const.drug_val))

            # get test data only with drugs
            df_test = self._get_df_with_reversed_edges(
                read_csv(self.data_path + cfg.data_const.drug_test))
        else:
            df_train = read_csv(self.data_path + cfg.data_const.drug_train)

            df_val = read_csv(self.data_path + cfg.data_const.drug_val)

            df_test = read_csv(self.data_path + cfg.data_const.drug_test)

        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)
        df_drug = concat([df_train, df_val, df_test]).reset_index(drop=True)

        self.ent_maps = read_csv(self.data_path + cfg.data_const.ent_maps)
        self.rel_maps = read_csv(self.data_path + cfg.data_const.rel_maps)
        
        df_train, df_val = self._add_proteins(df_train, df_val, cfg)

        return df_train, df_val, df_test, df_drug

    @staticmethod
    def _get_df_with_reversed_edges(df: DataFrame) -> DataFrame:
        """
        For dataframe with oriented triples return dataframe with original and
        reversed triples.

        Parameters
        ----------
        df : DataFrame
            Dataframe with oriented triples.

        Returns
        -------
        DataFrame :
            Dataframe with original and reversed triples.
        """
        reversed_data = df.copy(deep=True)
        cols = reversed_data.columns
        reversed_data = reversed_data[[cols[2], cols[1], cols[0]]]
        reversed_data.columns = cols
        return concat([df, reversed_data]).reset_index(drop=True)

    def _add_proteins(self, df_train, df_val, cfg: DictConfig):
        if self.use_proteins:
            # load ppi data
            max_drug_rel_id = self.rel_maps['id_in_data'].max()
            df_ppi_train, df_ppi_val = self._load_protein_data(
                cfg.data_const.ppi,
                self.use_prot_on_val,
                max_drug_rel_id + 1)

            # load targets
            df_tar_train, df_tar_val = self._load_protein_data(
                cfg.data_const.targets,
                self.use_prot_on_val, max_drug_rel_id + 2)

            df_train = concat(
                [df_train, df_ppi_train, df_tar_train]).reset_index(drop=True)
            df_val = concat([df_val, df_ppi_val, df_tar_val]).reset_index(
                drop=True)
        return df_train, df_val

    def _load_protein_data(self, csv_file: str, use_protein_in_validation: bool,
                           rel_id: int) -> Tuple[DataFrame, DataFrame]:
        df = read_csv(self.data_path + csv_file)
        cols = df.columns
        df[KG_CONST['column_names'][1]] = [rel_id] * len(df)
        df = df[
            [cols[0], KG_CONST['column_names'][1], cols[1]]]
        df.columns = KG_CONST['column_names']

        if self.use_reversed_edges:
            self.df_protein = self._get_df_with_reversed_edges(df)
        else:
            self.df_protein = df

        if use_protein_in_validation:
            len_train = int(len(df) * 0.8)
            df_train = self.df_protein.iloc[:len_train]
            df_val = self.df_protein.iloc[len_train:]
        else:
            df_train = self.df_protein
            df_val = self.df_protein.iloc[:0]

        return df_train, df_val

    def get_num_of_ent(self, data_type: str) -> int:
        """
        Get number of different entities types.

        Parameters
        ----------
        data_type : str
            Train, val or test.

        Returns
        -------
        int :
            Number of different entities types.

        Raises
        ------
        ValueError:
            If data_type not one of train, val or test.
        """
        if data_type == "train":
            df = concat([self._df_train, self._df_val, self._df_test])
            # add 1 to max(...), cause max index is smaller by 1
            return max(np.max(df[KG_CONST['column_names'][0]]),
                       np.max(df[KG_CONST['column_names'][2]])) + 1
        if data_type == "val":
            if self.use_prot_on_val:
                return self.get_num_of_ent("train")
            return len(self.ent_maps)
        if data_type == "test":
            return len(self.ent_maps)
        raise ValueError("Unknown data_type!")

    def get_num_of_rel(self, data_type: str):
        """
        Get number of different relatives types.

        Parameters
        ----------
        data_type : str
            Train, val or test.

        Returns
        -------
        int :
            Number of different relatives types.

        Raises
        ------
        ValueError:
            If data_type not one of train, val or test.
        """
        if data_type == "test":
            return len(self.rel_maps)
        if data_type == "train":
            if self.use_proteins:
                # additional types for prot-prot and drug-prot interaction
                return len(self.rel_maps) + 2
            return len(self.rel_maps)
        if data_type == "val":
            if self.use_prot_on_val:
                return len(self.rel_maps) + 2
            return len(self.rel_maps)
        raise ValueError("Unknown datatype!")

    def get_data_by_type(self, data_type: str):
        """
        Get data by type.

        Parameters
        ----------
        data_type : str
            Train, val or test.

        Returns
        -------
        DataFrame :
            Data by type.

        Raises
        ------
        ValueError:
            If data_type not one of train, val or test.
        """
        if data_type == "train":
            return self._df_train
        if data_type == "val":
            return self._df_val
        if data_type == "test":
            return self._df_test
        raise ValueError("Unknown datatype!")

    def get_drugs_array(self):
        """
        Returns all (drug, relation, drug) triples.

        Returns
        -------
        drugs_array : np.array
            Array of all (drug, relation, drug) triples.
        """
        return self._df_drug.values


class DrugsEdgesKnowledgeGraph(KnowledgeGraph):
    """
    EdgesDegreeKG with stored descriptors.

    Attributes
    ----------
    descriptors_type : str
        Type of descriptors used
    """
    def __init__(self, cfg: DictConfig, data_path, use_proteins,
                 use_proteins_on_validation,
                 use_reversed_edges: bool,
                 descriptors_type: str):
        KnowledgeGraph.__init__(self, cfg, data_path, use_proteins, use_proteins_on_validation, use_reversed_edges)
        self.descriptors = read_csv(data_path + cfg['embedding_data'][descriptors_type]['data'], index_col=0).values
        self.edges_degree = self._get_edges_degree()
        self.similarity_score = None

    def _get_edges_degree(self):
        degree_df = concat([self._df_train[KG_CONST['column_names'][0]],
                                   self._df_train[KG_CONST['column_names'][2]]]) \
            .value_counts(sort=False)

        edges_pwr = np.zeros(self.get_num_of_ent("train"))
        for idx in degree_df.index:
            edges_pwr[idx] = degree_df[idx]
        return edges_pwr.reshape(-1,1)

    def get_all_drug_pairs(self):
        """
        Get all possible pairs (drug_1, drug_2).
        Returns
        -------
        drugs_pairs : torch.Tensor
            tensor of size (num_of_drugs^2, 2).
        """
        num_of_drugs = self.get_num_of_ent('test')

        return torch.tensor([(i, j) for i in range(num_of_drugs)
                             for j in range(num_of_drugs)])

    @abstractmethod
    def calculate_pairwise_similarity(self):
        pass


class DegreeSimilarityKnowledgeGraph(DrugsEdgesKnowledgeGraph):
    """
    DrugsEdgesKnowledgeGraph with stored descriptors.

    Attributes
    ----------
    similarity_type : str
        Type of similarity function.
    """

    def __init__(self, cfg: DictConfig, data_path, use_proteins,
                 use_proteins_on_validation,
                 use_reversed_edges: bool, descriptors_type: str, similarity_type: str):
        DrugsEdgesKnowledgeGraph.__init__(self, cfg, data_path, use_proteins, use_proteins_on_validation,
                         use_reversed_edges, descriptors_type)
        self.similarity_score = get_embedding_similarity_by_type(similarity_type, self.descriptors)

    def calculate_nodes_degrees_matrix(self, get_edges_degree_score):
        """
        Calculate pairwise matrix, based on nodes degree.

        Parameters
        ----------
        get_edges_degree_score : Callable
            Function that measures similarity of 2 nodes, based on nodes degree.

        Returns
        -------
        pairwise_degree_similarity : array-like (2d)
        """
        edges_degree_score = get_edges_degree_score(self.edges_degree)
        pairwise_degree_similarity = pairwise_distances(self.edges_degree,
                                                 metric=edges_degree_score)

        return pairwise_degree_similarity

    def calculate_pairwise_similarity(self):
        pairwise_similarity = pairwise_distances(self.descriptors,
                                                 metric=self.similarity_score)

        return pairwise_similarity


class ParzenSimilarityKnowledgeGraph(DrugsEdgesKnowledgeGraph):
    """
    Knowledge graph with weighted edges, weight counted using Parzen window estimation.

    Attributes
    ----------

    """

    def __init__(self, cfg: DictConfig, data_path, use_proteins,
                 use_proteins_on_validation,
                 use_reversed_edges: bool, descriptors_type: str):
        DrugsEdgesKnowledgeGraph.__init__(self, cfg, data_path, use_proteins, use_proteins_on_validation,
                         use_reversed_edges, descriptors_type)
        self.min_degree = self.edges_degree.min()
        self.max_degree = self.edges_degree.max()
        self.similarity_score = lambda x, y: self._parzen_similarity_score(x,y)

    def _parzen_similarity_score(self, fp1_idx, fp2_idx):
        fp1_idx = int(fp1_idx[0])
        fp2_idx = int(fp2_idx[0])
        fp1_degree = 1 - (self.edges_degree[fp1_idx] - self.min_degree) / (self.max_degree - self.min_degree)
        fp2_degree = 1 - (self.edges_degree[fp2_idx] - self.min_degree) / (self.max_degree - self.min_degree)
        sigma = PARZEN_WINDOW_PARAMS['window_bounds'][0] + fp1_degree * fp2_degree *\
                (PARZEN_WINDOW_PARAMS['window_bounds'][1] - PARZEN_WINDOW_PARAMS['window_bounds'][0])
        return exp_similarity(self.descriptors[fp1_idx], self.descriptors[fp2_idx], sigma=sigma)

    def calculate_pairwise_similarity(self):
        pairwise_similarity = pairwise_distances(np.arange(0, self.descriptors.shape[0]).reshape(-1, 1),
                                                 metric=self.similarity_score)

        return pairwise_similarity


class MSEKnowledgeGraph(KnowledgeGraph):
    def __init__(self, cfg: DictConfig, data_path, use_proteins,
                 use_proteins_on_validation,
                 use_reversed_edges: bool, use_saves: bool,
                 n_closest: int, weak_node_list: Optional[str]):
        KnowledgeGraph.__init__(self, cfg, data_path, use_proteins, use_proteins_on_validation,
                         use_reversed_edges)
        self.n_closest = n_closest
        if use_saves:
            if os.path.exists(cfg.data_const.work_dir + cfg.data_const.nearest_drugs):
                with open(cfg.data_const.work_dir + cfg.data_const.nearest_drugs, 'rb') as f:
                    self.nearest = pickle.load(f)
                print("Nearest nodes loaded!")
            else:
                print("Nearest nodes saves doesn't found. Computing...")
                self.nearest = self._get_nearest(cfg, True)
        else:
            self.nearest = self._get_nearest(cfg)
        nodes_in_train = set(concat([self._df_train[KG_CONST['column_names'][0]],
                                 self._df_train[KG_CONST['column_names'][2]]]))

        if weak_node_list is not None:
            with open(weak_node_list, 'rb') as f:
                self.weak_nodes = pickle.load(f)
        else:
            self.weak_nodes = list(set(range(len(self.ent_maps))).difference(nodes_in_train))

    def _get_nearest(self, cfg: DictConfig, use_saves: bool = False) -> list:
        def get_n_closest_drugs(se_one_hot, drug_id:int, n_closest: int):
            dist = {}
            try:
                non_zero_fxd = set(se_one_hot.loc[drug_id][se_one_hot.loc[drug_id] > 0].index)
                for i in se_one_hot.index:
                    non_zero_fst = set(se_one_hot.loc[i][se_one_hot.loc[i] > 0].index)
                    intersec = len(non_zero_fxd.intersection(non_zero_fst))
                    dist[i] = intersec/len(non_zero_fxd)
                dist = [k for k, v in sorted(dist.items(), key=lambda item: item[1], reverse=True) if v > 0]
                n_closest = min(n_closest, len(dist)-1)
                return dist[1:n_closest+1]
            except Exception:
                return []

        ent_maps = read_csv(cfg.data_const.work_dir + cfg.data_const.ent_maps)
        ent_maps.columns = ["STITCH", "ID"]

        num_of_drugs = len(ent_maps)

        mono_se = read_csv(cfg.data_const.work_dir + cfg.data_const.mono_se)
        mono_se = mono_se.drop(columns=['Side Effect Name']).set_index('STITCH')
        mono_se.columns = ['se']

        df = get_dummies(mono_se.se, prefix='')
        se_one_hot = df.groupby(['STITCH']).agg(lambda x: sum(x) if len(x) > 1 else x.iloc[0])
        se_one_hot = se_one_hot.merge(ent_maps, how='inner', on='STITCH').set_index('ID').drop(columns=['STITCH']).sort_index()
        closest_node = []
        for node in range(num_of_drugs):
            closest_node.append(get_n_closest_drugs(se_one_hot, node, self.n_closest))

        if use_saves:
            with open(cfg.data_const.work_dir + cfg.data_const.nearest_drugs, 'wb') as f:
                pickle.dump(closest_node, f)
        return closest_node


class MSEParzenKnowledgeGraph(ParzenSimilarityKnowledgeGraph, MSEKnowledgeGraph):
    def __init__(self, cfg: DictConfig, data_path, use_proteins,
                 use_proteins_on_validation,
                 use_reversed_edges: bool, descriptors_type: str, use_saves: bool,
                 n_closest: int, weak_node_list):
        ParzenSimilarityKnowledgeGraph.__init__(self, cfg, data_path, use_proteins,
                         use_proteins_on_validation,
                         use_reversed_edges, descriptors_type)
        MSEKnowledgeGraph.__init__(self, cfg, data_path, use_proteins,
                 use_proteins_on_validation,
                 use_reversed_edges, use_saves,
                 n_closest, weak_node_list)


class MSEDegreeSimilarityKnowledgeGraph(DegreeSimilarityKnowledgeGraph, MSEKnowledgeGraph):
    def __init__(self, cfg: DictConfig, data_path, use_proteins,
                 use_proteins_on_validation,
                 use_reversed_edges: bool, descriptors_type: str, similarity_type: str,
                 use_saves: bool,n_closest: int, weak_node_list):
        DegreeSimilarityKnowledgeGraph.__init__(self, cfg, data_path, use_proteins,
                         use_proteins_on_validation,
                         use_reversed_edges, descriptors_type, similarity_type)
        MSEKnowledgeGraph.__init__(self, cfg, data_path, use_proteins,
                                                use_proteins_on_validation,
                                                use_reversed_edges, use_saves,
                                                n_closest, weak_node_list)
