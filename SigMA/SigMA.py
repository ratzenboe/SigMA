import numpy as np
import pandas as pd
from SigMA.Resampling import PerturbedData
from SigMA.Parameters import ParameterClass
from collections import defaultdict
from itertools import groupby


class SigMA(ParameterClass, PerturbedData):
    def __init__(self,
                 data: pd.DataFrame,
                 cluster_features: list,
                 scale_factors: dict,
                 nb_resampling: int = 20,
                 max_knn_density: int = 100,
                 beta: float = 0.99,
                 knn_initcluster_graph: int = 70):
        """High-level class applying the SigMA_v0 clustering analysis
        data: pandas data frame
        cluster_features: Features used in the clustering process
        scale_factors: Features that are scaled with given factors, needs specific layout:
            scale_factors: {
                    pos: {'features': ['v_alpha', 'v_delta'], 'factor': 5}
            }
        ---
        max_knn_density: Maximum k for scale space density estimation
        ---
        beta: value for the beta-skeleton
        knn_initcluster_graph: maximum number of neighbors a node can have in the beta-skeleton
        """
        # Check if data is pandas dataframe or series
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError('Data input needs to be pandas DataFrame!')

        super().__init__(
            # DataLayer class attributes
            data=data, cluster_features=cluster_features, scale_factors=scale_factors,
            # DensityEstimator
            max_knn_density=max_knn_density,
            # GraphSkeleton
            knn_initcluster_graph=knn_initcluster_graph, beta=beta
        )
        # Resampling info
        self.nb_resampling = nb_resampling
        if nb_resampling > 0:
            self.build_covariance_matrix()
            print('Creating k-d trees of resampled data sets...')
            self.resampled_kdtrees = [self.create_kdtree() for i in range(self.nb_resampling)]

        # Saddle point information
        self.saddle_dknn = defaultdict(frozenset)
        self.cluster_saddle_points = defaultdict(frozenset)
        self.saddles_per_modes_density = None
        self.mode_neighbor_dict = None

    def initialize_mode_neighbor_dict(self):
        edges = []
        for e, c in self.saddle_dknn.keys():
            edges.append((e, c))
            edges.append((c, e))   # groupby only works in "one direction" so we store both edges
        self.mode_neighbor_dict = {k: np.array([v[1] for v in g]) for k, g in groupby(sorted(edges), lambda e: e[0])}
        return

    def mode_neighbors(self, mode_id: int):
        """Function needed for noise removal"""
        return self.mode_neighbor_dict[mode_id]

    def input_check(self, knn):
        if (self.knn != knn) or (self.knn is None):
            self.initialize_clustering(knn=knn)
        return

    def fit(self, alpha: float, knn: int = None, hypotest='hmp', saddle_point_candidate_threshold: int = 20):
        """
        :param alpha: significance level
        :param knn: knn density estimation parameter
        :param saddle_point_candidate_threshold: Number of candidates to consider when searching for saddle point
        """
        # Determine crucial variables
        if knn != self.knn:
            # Need to perform gradient ascend step
            self.initialize_clustering(knn=knn, saddle_point_candidate_threshold=saddle_point_candidate_threshold)
            # Compute look-up dictionary, i.e. adjacency list, for fast mode_neighbors function
            self.initialize_mode_neighbor_dict()
            # Compute k-distances for re-sampled data sets
            self.resample_k_distances()

        labels = self.merge_clusters(knn=knn, alpha=alpha, hypotest=hypotest)
        return labels

    def resample_k_distances(self):
        # Loop through all trees and compute k-distances on resampled data sets
        for kd_tree_i in self.resampled_kdtrees:
            mode_list = np.array(list(self.mode_kdist.keys()))
            dists, _ = kd_tree_i.query(self.X[mode_list], k=self.knn + 1, n_jobs=-1)
            kdist_modes = np.max(dists, axis=1)
            # ------- Modes ----------
            # add distances to mode list
            for mode_i, kdist_i in zip(mode_list, kdist_modes):
                self.mode_kdist[mode_i].append(kdist_i)
            # ------- Saddles --------
            all_saddle_points = self.saddle_point_positions()
            dists, _ = kd_tree_i.query(all_saddle_points, k=self.knn + 1, n_jobs=-1)
            kdist_saddles = np.max(dists, axis=1)
            for (saddle_kdist_list, _), kdist_i in zip(self.saddle_dknn.values(), kdist_saddles):
                saddle_kdist_list.append(kdist_i)
        return
