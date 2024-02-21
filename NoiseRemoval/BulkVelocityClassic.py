import numpy as np
from astropy import units as u
from astropy.coordinates import Galactic, CartesianDifferential, ICRS
from scipy import stats
from scipy.optimize import minimize
from NoiseRemoval.gagne_helper_functions import equatorial_XYZ
from NoiseRemoval.BulkVelocitySolverBase import VelocityEstimatorBase


class ClassicBV(VelocityEstimatorBase):
    def __init__(self, data):
        """Class to compute the bulk velocity of a single cluster by minimizing total mahanobolis distance to a common center.
        For now, this class assumes Gaia feature names!
        """
        super().__init__(data=data)

    def initial_guess(self, data_subset, cluster_subset=None):
        rv_not_nan = self._subset_handler(~self.rv_isnan, cluster_subset)
        data_subset_notnan = data_subset.loc[rv_not_nan]
        if np.sum(rv_not_nan) <= 5:
            return np.ones(3)

        # Get astronometric data
        ra, dec, plx, pmra, pmdec, rv = data_subset_notnan[
            ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']].values.T
        # Compute mean UVW
        UVW = self.err_sampler.spher2cart(np.vstack((ra, dec, plx, pmra, pmdec, rv)).T)[:, 3:]
        return np.median(UVW, axis=0)

    def fit(self, cluster_subset=None, method='BFGS'):
        data_subset = self.data.loc[self._subset_handler(cluster_subset)]
        cols = ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity', 'pmra_error', 'pmdec_error',
                'radial_velocity_error']
        ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err = data_subset[cols].values.T
        # Transform data to XYZ
        X_hat, Y_hat, Z_hat = equatorial_XYZ(ra, dec, 1000 / plx)
        cos_dec = np.cos(np.radians(dec))
        # Transform data with astropy SkyCoord to ICRS
        gal_coords = Galactic(
            u=X_hat * u.pc,
            v=Y_hat * u.pc,
            w=Z_hat * u.pc,
            # velocities UVW
            U=np.zeros_like(X_hat) * u.km / u.s,
            V=np.zeros_like(X_hat) * u.km / u.s,
            W=np.zeros_like(X_hat) * u.km / u.s,
            representation_type="cartesian",
            # Velocity representation
            differential_type="cartesian",
        )
        # Initial guess
        v_guess = self.initial_guess(data_subset, cluster_subset)
        # Minimize
        res = minimize(
            fun=self.ll_skycoods,
            x0=np.r_[v_guess, np.array([2])],
            args=(gal_coords, cos_dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err),
            method=method
        )
        return res

    @staticmethod
    def ll_skycoods(
            theta, gal_coords, cos_dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err
    ):
        # Cluster dispersion important for updated log likelihood calculation
        v1, v2, v3, cluster_disp = theta
        # Update velocities
        gal_coords.data.differentials["s"] = CartesianDifferential(
            d_x=np.full_like(plx, fill_value=v1) * u.km / u.s,
            d_y=np.full_like(plx, fill_value=v2) * u.km / u.s,
            d_z=np.full_like(plx, fill_value=v3) * u.km / u.s,
        )
        # transform to ICRS
        icrs_coords = gal_coords.transform_to(ICRS())
        icrs_coords.representation_type = "spherical"

        # Get proper motions and radial velocity
        pmra_calc = icrs_coords.pm_ra.value * cos_dec
        pmdec_calc = icrs_coords.pm_dec.value
        rv_calc = icrs_coords.radial_velocity.value
        # compute log likelihood
        ll_rv = stats.t.logpdf(
            x=rv_calc, df=1, loc=rv, scale=np.sqrt(rv_err ** 2 + cluster_disp ** 2)
        )
        ll_pmra = stats.t.logpdf(
            x=pmra_calc, df=1, loc=pmra, scale=np.sqrt(pmra_err ** 2 + cluster_disp ** 2)
        )
        ll_pmdec = stats.t.logpdf(
            x=pmdec_calc, df=1, loc=pmdec, scale=np.sqrt(pmdec_err ** 2 + cluster_disp ** 2)
        )
        return -np.sum(ll_pmra + ll_pmdec + ll_rv)
