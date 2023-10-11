import numpy as np
import nglpy
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.stats import norm
from scipy.sparse import csr_matrix
from SigMA.DensityEstimator import DensityEstimator


def compute_pvalue(k, p, d_max, d_saddle):
    SB_alpha = p * np.sqrt(k / 2) * (np.log(d_saddle) - np.log(d_max))
    return 1 - norm.cdf(SB_alpha)


class GraphSkeleton(DensityEstimator):
    def __init__(self, knn_initcluster_graph: int, beta: float, **kwargs):
        """
        :param max_neighbors: maximal number of neighbors considered as graph edges
        :param beta: structure parameter of beta skeleton
        :param knn: Number of neighbors for
        """
        super().__init__(**kwargs)
        self.knn_initcluster_graph = knn_initcluster_graph
        self.beta = beta
        self.A = None
        self.setup()

    def setup(self):
        # If metric_params are given, we build the graph with the given metric
        if self.metric_params:
            # Should be passed in the following way:
            # metric_params = dict(metric='mahalanobis', metric_params={'V': np.cov(X_cluster.T)}, **other_kwargs)
            self.A = kneighbors_graph(self.X, self.knn_initcluster_graph, n_jobs=-1, **self.metric_params)
        # If beta is given, we build the beta skeleton
        elif isinstance(self.beta, (float, int)):
            self.A = self.beta_adjacency(max_neighbors=self.knn_initcluster_graph, beta=self.beta)
            # self.A = self.remove_edges()
        else:
            self.A = kneighbors_graph(self.X, self.knn_initcluster_graph, n_jobs=-1)

    def beta_adjacency(self, max_neighbors: int, beta: float):
        # Build beta skeleton
        erg = nglpy.EmptyRegionGraph(max_neighbors=max_neighbors, relaxed=True, beta=beta)
        erg.build(self.X)
        rows, cols = [], []
        for i, (node, edges) in enumerate(erg.neighbors().items()):
            rows.extend(len(edges) * [node])
            cols.extend(list(edges))
        return csr_matrix((np.ones_like(rows), (rows, cols)), shape=(self.X.shape[0], self.X.shape[0]))

    def update_scale_factors(self, scale_factors: dict, metric_params: dict = dict()):
        """Scale factor update inn Density layer as distances also need updating"""
        self.scale_factors = scale_factors
        if self.metric_params != metric_params:
            self.metric_params = metric_params
        # Update X
        self.X = self.init_cluster_data()
        # Update kd-tree
        # self.kd_tree = cKDTree(data=self.X)
        self.kd_tree = NearestNeighbors(n_jobs=-1, **self.metric_params).fit(self.X)
        # Update distances
        self.distances = self.calc_distances()
        # Build graph
        self.setup()
        return

    # def remove_edges(self):
    #     # We remove edges where the mid-points between 2 vertices has a significant dip
    #     # --> we choose 7 neighbors for this test to be aware of local changes
    #     knn_test = 7
    #     e1, e2 = self.A.nonzero()
    #     midpts = (self.X[e1] + self.X[e2]) * 0.5
    #     dists_midpoints, _ = self.kd_tree.query(midpts, k=knn_test, workers=-1)
    #     d_saddle = np.max(dists_midpoints, axis=1)
    #     dists = self.k_distance(knn_test)
    #     d_max = np.max(np.vstack([dists[e1], dists[e2]]), axis=0)
    #     pvals = compute_pvalue(k=knn_test, p=self.X.shape[-1], d_max=d_max, d_saddle=d_saddle)
    #     keep_edges = pvals > 0.05  # 5% significance level
    #     rows, cols = e1[keep_edges], e2[keep_edges]
    #     A = csr_matrix((np.ones_like(rows), (rows, cols)), shape=(self.X.shape[0], self.X.shape[0]))
    #     return A
