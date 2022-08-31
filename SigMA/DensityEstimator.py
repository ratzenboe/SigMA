import numpy as np
from SigMA.DataLayer import DataLayer


class DensityEstimator(DataLayer):
    def __init__(self, max_knn_density: int = None, **kwargs):
        """Class calculating densities on given data X
        max_knn_density: maximal neighbors to consider in knn density estimation
        """
        super().__init__(**kwargs)
        self.max_knn_density = max_knn_density
        self.distances = self.calc_distances()

    def calc_distances(self):
        dists = None
        if (self.data is not None) and (self.max_knn_density is not None):
            dists, _ = self.kd_tree.query(self.X, k=self.max_knn_density+1, n_jobs=-1)
            dists = np.sort(dists[:, 1:], axis=1)
        return dists

    def knn_density(self, k_neighbors: int):
        if self.max_knn_density < k_neighbors:
            raise ValueError('Given k_neighbors is larger than max_neighbors')
        return 1 / np.sqrt(np.mean(np.square(self.distances[:, :k_neighbors-1]), axis=1))

    def k_distance(self, k: int):
        return self.distances[:, k-1]
