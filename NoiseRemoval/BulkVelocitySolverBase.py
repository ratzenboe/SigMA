import numpy as np
import copy
from NoiseRemoval.OptimalVelocity import vr_solver
from miscellaneous.error_sampler import ErrorSampler
from sklearn.covariance import MinCovDet


class VelocityEstimatorBase:
    def __init__(self, data):
        # Most important: store NaN information --> should be index array for easier handling
        self.rv_isnan = data['radial_velocity'].isna().values.ravel()
        self.data = self.set_data(data)
        self.data_idx = np.arange(data.shape[0])
        self.err_sampler = ErrorSampler()

    def _subset_handler(self, cluster_subset, bool_array=None):
        # Type checks
        if cluster_subset is not None:
            if cluster_subset.dtype != bool:
                cluster_subset = np.isin(self.data_idx, cluster_subset)
        if bool_array is not None:
            if bool_array.dtype != bool:
                bool_array = np.isin(self.data_idx, bool_array)

        # Return subset
        if cluster_subset is None:
            if bool_array is not None:
                return bool_array
            else:
                return self.data_idx
        else:
            if bool_array is not None:
                return cluster_subset[bool_array]
            else:
                return cluster_subset

    def set_data(self, data):
        data = copy.deepcopy(data)
        data.loc[self.rv_isnan, 'radial_velocity_error'] = 1e3
        data.loc[self.rv_isnan, 'radial_velocity'] = 0.0
        return data

    def estimate_rv(self, cluster_subset=None, return_full=False, **kwargs):
        # Estimate mean UVW
        res = self.fit(cluster_subset, **kwargs)
        # Get estimated UVW
        U, V, W = res.x[:3]
        # Get observed data
        cols_sphere = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
        ra, dec, plx, pmra, pmdec = self.data.loc[self._subset_handler(cluster_subset), cols_sphere].values.T
        # Estimate radial velocity
        vr_est = vr_solver(U, V, W, ra, dec, plx, pmra, pmdec)
        # Get available radial velocity measurements
        rv = self.data.loc[self._subset_handler(cluster_subset), 'radial_velocity'].values
        rv_is_nan_cluster = self._subset_handler(self.rv_isnan, cluster_subset)
        rv[rv_is_nan_cluster] = vr_est[rv_is_nan_cluster]
        # Return either full information or only radial velocities
        if return_full:
            return ra, dec, plx, pmra, pmdec, rv
        else:
            return rv

    def estimate_uvw(self, cluster_subset=None, **kwargs):
        # Estimate mean UVW
        ra, dec, plx, pmra, pmdec, rv = self.estimate_rv(cluster_subset, return_full=True, **kwargs)
        # Compute UVW
        return self.err_sampler.spher2cart(np.vstack((ra, dec, plx, pmra, pmdec, rv)).T)[:, 3:]

    def estimate_normal_params(self, cluster_subset=None, **kwargs):
        support_fraction = kwargs.pop('support_fraction', None)
        UVW = self.estimate_uvw(cluster_subset, **kwargs)
        mcd = MinCovDet(support_fraction=support_fraction).fit(UVW)
        return mcd.location_, mcd.covariance_
