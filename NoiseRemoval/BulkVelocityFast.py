import numpy as np
from miscellaneous.error_sampler import ErrorSampler
from scipy.optimize import minimize
from NoiseRemoval.BulkVelocitySolverBase import VelocityEstimatorBase


class FastBV(VelocityEstimatorBase):
    def __init__(self, data):
        """Class to compute the bulk velocity of a single cluster by minimizing total mahanobolis distance to a common center.
        For now, this class assumes Gaia feature names!
        """
        super().__init__(data=data)
        # Set member variables
        self.X = None
        self.C = None
        self.set_mem_vars()

    def set_mem_vars(self):
        self.err_sampler = ErrorSampler(self.data)
        self.err_sampler.build_covariance_matrix()
        # Sample from errors
        X_sphere = self.err_sampler.new_sample()
        self.X = self.err_sampler.spher2cart(X_sphere)[:, 3:]
        self.C = self.err_sampler.C[:, 3:, 3:]
        return

    def fit(self, cluster_subset=None):
        X_fit = self.X[self._subset_handler(cluster_subset)]
        C_fit = self.C[self._subset_handler(cluster_subset)]
        # Initial guess
        rv_not_nan = self._subset_handler(~self.rv_isnan, cluster_subset)
        if np.sum(rv_not_nan) >= 5:
            v_guess = np.median(X_fit[rv_not_nan], axis=0)
        else:
            v_guess = np.ones(3)
        # Minimize
        res = minimize(self.minimizer_lts, v_guess, args=(X_fit, C_fit), method='Nelder-Mead')
        return res

    def minimizer_lts(self, theta, X, C):
        # Exp because entries need to be positive
        mahas = self.multiple_mahas(theta, X, C)
        # Get rid of 30 percent of the largest mahalanobis distances
        mahas = np.sort(mahas)
        mahas = mahas[:int(0.7 * len(mahas))]
        return np.sum(mahas)

    @staticmethod
    def multiple_mahas(x, means, covs):
        """Fast computation of multivariate normal log PDF over multiple sets of covariances.
        Courtesy of: https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
        means can be a 1D or 2D array, covs must be a 3D array.
        """
        # NumPy broadcasts `eigh`.
        vals, vecs = np.linalg.eigh(covs)
        # # Compute the log determinants across the second axis.
        # logdets = np.sum(np.log(vals), axis=1)
        # Invert the eigenvalues
        valsinvs = 1. / vals
        # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
        Us = vecs * np.sqrt(valsinvs)[:, None]
        devs = x - means
        # Use `einsum` for matrix-vector multiplications across the first dimension.
        devUs = np.einsum('ni,nij->nj', devs, Us)
        # Compute the Mahalanobis distance by squaring each term and summing.
        mahas = np.sum(np.square(devUs), axis=1)
        return mahas
