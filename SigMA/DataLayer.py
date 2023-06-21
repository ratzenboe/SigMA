import copy
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


class DataLayer:
    def __init__(self,
                 data: pd.DataFrame, cluster_features: list, scale_factors: dict, metric_params: dict = dict(),
                 **kwargs):
        """Class calculating densities on given data X
        data: pandas data frame containing all necessary columns
        cluster_features: Features used in the clustering process
        scale_factors: Features that are scaled with given factors, needs specific layout:
            scale_factors: {
                    pos: {'features': ['v_alpha', 'v_delta'], 'factor': 5}
            }

        """
        self.data = data
        self.cluster_columns = cluster_features
        self.scale_factors = scale_factors
        self.metric_params = metric_params
        self.X = self.init_cluster_data()
        # self.kd_tree = cKDTree(data=self.X)
        # Need NeareatNeighbors for Mahalanobis distance (not implemented in cKDTree)
        self.kd_tree = NearestNeighbors(n_jobs=-1, **metric_params).fit(self.X)


    def update_scaling_factors(self, scale_factors: dict, metric_params: dict = None):
        """Change scale factors and re-initialize clustering"""
        self.scale_factors = scale_factors
        # Update data and kd-tree
        self.X = self.init_cluster_data()
        # self.kd_tree = cKDTree(data=self.X)
        if not isinstance(metric_params, dict):
            metric_params = self.metric_params
        self.kd_tree = NearestNeighbors(n_jobs=-1, **metric_params).fit(self.X)
        return

    def init_cluster_data(self):
        X = copy.deepcopy(self.data[self.cluster_columns])
        for scale_info in self.scale_factors.values():
            cols = scale_info['features']
            sf = scale_info['factor']
            X[cols] *= sf
        return X.values
