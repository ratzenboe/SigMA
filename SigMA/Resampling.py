import numpy as np
import pandas as pd
import copy
from astropy.coordinates import LSR, SkyCoord, Distance
import astropy.units as u
from numba import jit
from SigMA.DataLayer import DataLayer
from scipy.spatial import cKDTree


class PerturbedData(DataLayer):
    def __init__(self, **kwargs):
        """Creating perturbed data sampled from normal density centered on data points
        data: data containing ra, dec, parallax, pmra, pmdec
              and corresponding error features + correlations of errors for covariance matrix
        """
        super().__init__(**kwargs)
        # Define necessary features
        self.astrometric_features = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
        self.astrometric_errors = ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']
        # Correlation positions in correlation/covariance matrix
        self.corr_map = {
            'ra_dec_corr': [0, 1],
            'ra_parallax_corr': [0, 2],
            'ra_pmra_corr': [0, 3],
            'ra_pmdec_corr': [0, 4],
            'dec_parallax_corr': [1, 2],
            'dec_pmra_corr': [1, 3],
            'dec_pmdec_corr': [1, 4],
            'parallax_pmra_corr': [2, 3],
            'parallax_pmdec_corr': [2, 4],
            'pmra_pmdec_corr': [3, 4]
        }
        # Data and cov. storage
        self.C = None  # covariance matrix for "astrometric_features" features
        self.L = None  # Cholesky decomposition, needed for fast sampling with numba
        self.X_orig = None   # assign variable only if needed --> i.e. in build_covariance_matrix function
        self.resampled_kdtrees = []  # list of kd trees the resampling has generated

    def build_covariance_matrix(self):
        """Create covariance matrix from input features"""
        # Assign X_orig
        self.X_orig = self.data[self.astrometric_features].values
        # Start building covariance matrix
        nb_points, nb_covfeats = self.X.shape[0], len(self.astrometric_features)
        # Initialize empty covariance matrix
        self.C = np.zeros((nb_points, nb_covfeats, nb_covfeats), dtype=np.float32)
        diag = np.arange(nb_covfeats)  # indices for diagonal
        # Fill diagonal with error columns
        self.C[:, diag, diag] = self.data[self.astrometric_errors].fillna(1e6).to_numpy(dtype=np.float32)
        # Fill in off-diagonal elements
        for column, (i, j) in self.corr_map.items():
            self.C[:, i, j] = self.data[column].fillna(0).to_numpy(dtype=np.float32)
            self.C[:, i, j] *= (self.C[:, i, i] * self.C[:, j, j])  # transform correlation to covariance
            self.C[:, j, i] = self.C[:, i, j]  # fill in symmetric component
        # Squre variance -> std dev
        self.C[:, diag, diag] = self.C[:, diag, diag] ** 2
        # Compute Cholesky decomposition
        epsilon = 0.0001  # Define epsilon to add small pertubation to
        self.L = copy.deepcopy(self.C)
        # add small pertubation to covariance matrix, because its eigenvalues can decay
        # very rapidly and without this stabilization the Cholesky decomposition fails
        self.L[:, diag, diag] += epsilon
        #  Cholesky decomposition.
        for k in range(nb_points):
            # set i'th element to Cholensky decomposition of covariance matrix
            self.L[k, :, :] = np.linalg.cholesky(self.L[k, :, :])

    def new_sample(self):
        nb_points, nb_covfeats = self.X.shape[0], len(self.astrometric_features)
        add2X = self.new_sample_jit(self.L, nb_points, nb_covfeats)
        X_new = self.X_orig + np.asarray(add2X)
        # Check for negative parallax cases, fix them
        neg_parallax = X_new[:, 2] <= 0
        if np.count_nonzero(neg_parallax) > 0:
            print('Negative parallax values encountered, fixing values...')
            X_new[neg_parallax] = self.resample_until_pos_plx(neg_parallax)
        # Check if the declination angle is out of bounds, if so we fix that
        if np.count_nonzero(np.abs(X_new[:, 1]) > 90) > 0:
            print('Dec out of bounds: Performing correction')
            X_new[:, 0], X_new[:, 1] = self.transform_to_radec_bounds(ra=X_new[:, 0], dec=X_new[:, 1])
        return X_new

    @staticmethod
    @jit(nopython=True)
    def new_sample_jit(L, nb_points, nb_covfeats):
        """Sample a single data point from normal distribution
        Here we calculate the distance we have to travel away from data point:
        X'[i] = X[i] + sample_list[i]
        """
        sample_list = list()
        for i in range(nb_points):
            u = np.random.normal(loc=0, scale=1, size=nb_covfeats).astype(np.float32)
            mult = L[i] @ u
            sample_list.append(mult)
        return sample_list

    def transform_to_radec_bounds(self, ra, dec):
        # skycoords can deal with ra>360 or ra<0, but not with dec out of [-90, 90]
        dec_under_range = dec < -90.
        dec_over_range = dec > 90.
        # Mirror the declination angle
        dec[dec_under_range] += 2 * (np.abs(dec[dec_under_range]) - 90)
        dec[dec_over_range] -= 2 * (np.abs(dec[dec_over_range]) - 90)
        # We have passed the poles, so we need to turn ra by 180 degrees
        ra[dec_under_range | dec_over_range] += 180
        return ra, dec

    def resample_until_pos_plx(self, neg_parallax):
        samples_list = list()
        for i in np.flatnonzero(neg_parallax):
            sample = np.random.multivariate_normal(self.X_orig[i], self.C[i, :, :], size=1).flatten()
            # There shouldn't be any parallax values smaller than 0!
            while sample[2] <= 0.:
                # Create new sample until
                sample = np.random.multivariate_normal(self.X_orig[i], self.C[i, :, :], size=1).flatten()
            samples_list.append(sample)
        return np.array(samples_list)

    def change_coordinates(self, sampled_data):
        # Transform to different coordinate system
        skycoord = SkyCoord(
            ra=sampled_data[:, 0] * u.deg,
            dec=sampled_data[:, 1] * u.deg,  # 2D on sky postition
            distance=Distance(parallax=sampled_data[:, 2] * u.mas),  # distance in pc
            pm_ra_cosdec=sampled_data[:, 3] * u.mas / u.yr,
            pm_dec=sampled_data[:, 4] * u.mas / u.yr,
            radial_velocity=0. * u.km / u.s,
            frame="icrs"
        )
        x = skycoord.galactic.cartesian.x.value
        y = skycoord.galactic.cartesian.y.value
        z = skycoord.galactic.cartesian.z.value
        # Transform to lsr
        pma_lsr = skycoord.transform_to(LSR()).pm_ra_cosdec.value
        pmd_lsr = skycoord.transform_to(LSR()).pm_dec.value
        v_a_lsr = 4.74047 * pma_lsr / sampled_data[:, 2]
        v_d_lsr = 4.74047 * pmd_lsr / sampled_data[:, 2]
        return pd.DataFrame({'X': x, 'Y': y, 'Z': z, 'v_a_lsr': v_a_lsr, 'v_d_lsr': v_d_lsr})

    def create_kdtree(self):
        # Resample data set and compute saddle point densities
        X_new = self.new_sample()
        data_resampled = self.change_coordinates(X_new)
        # Scale data
        data_resampled = data_resampled[self.cluster_columns]
        for scale_info in self.scale_factors.values():
            cols = scale_info['features']
            sf = scale_info['factor']
            data_resampled[cols] *= sf
        kd_tree_resample = cKDTree(data=data_resampled.values)
        return kd_tree_resample
