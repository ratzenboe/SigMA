import numpy as np
from NoiseRemoval.gagne_helper_functions import equatorial_XYZ
from astropy.coordinates import Galactic, CartesianDifferential, ICRS
from astropy import units as u
from scipy import optimize, stats


def dense_sample(rho):
    """Extract the densest points from the density distribution."""
    mad = np.median(np.abs(rho - np.median(rho)))
    threshold = np.median(rho) * 0.995 + 3 * mad * 1.1
    if np.sum(rho > threshold) < 20:
        threshold = np.percentile(rho, 93)
    return rho > threshold


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
        x=rv_calc, df=1, loc=rv, scale=np.sqrt(rv_err**2 + cluster_disp**2)
    )
    ll_pmra = stats.t.logpdf(
        x=pmra_calc, df=1, loc=pmra, scale=np.sqrt(pmra_err**2 + cluster_disp**2)
    )
    ll_pmdec = stats.t.logpdf(
        x=pmdec_calc, df=1, loc=pmdec, scale=np.sqrt(pmdec_err**2 + cluster_disp**2)
    )
    return -np.sum(ll_pmra + ll_pmdec + ll_rv)


def optimize_velocity_skycoords(
    ra,
    dec,
    plx,
    pmra,
    pmdec,
    rv,
    pmra_err,
    pmdec_err,
    rv_err,
    init_guess=np.ones(4),
    method="BFGS",
    **kwargs
):
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

    sol = optimize.minimize(
        # fun=transform_inverse,
        fun=ll_skycoods,
        x0=init_guess,
        method=method,
        args=(gal_coords, cos_dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err),
        **kwargs
    )
    return sol


def bulk_velocity_solver(
    data,
    rho,
    ra_col="ra",
    dec_col="dec",
    plx_col="parallax",
    pmra_col="pmra",
    pmdec_col="pmdec",
    rv_col="radial_velocity",
    pmra_err_col="pmra_error",
    pmdec_err_col="pmdec_error",
    rv_err_col="radial_velocity_error",
    **kwargs
):
    # Get the densest points
    cut_dense_core = dense_sample(rho)
    # Get the densest points
    cols = [
        ra_col,
        dec_col,
        plx_col,
        pmra_col,
        pmdec_col,
        rv_col,
        pmra_err_col,
        pmdec_err_col,
        rv_err_col,
    ]
    ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err = data.loc[
        cut_dense_core
    ][cols].values.T
    # Replace nans
    rv[np.isnan(rv)] = 0
    rv_err[np.isnan(rv_err)] = 1e5
    # Compute optimal velocity
    sol = optimize_velocity_skycoords(
        ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err, **kwargs
    )
    return sol


def bootstrap_bulk_velocity_solver(
    data,
    rho,
    n_bootstraps=100,
    ra_col="ra",
    dec_col="dec",
    plx_col="parallax",
    pmra_col="pmra",
    pmdec_col="pmdec",
    rv_col="radial_velocity",
    pmra_err_col="pmra_error",
    pmdec_err_col="pmdec_error",
    rv_err_col="radial_velocity_error",
    **kwargs
):
    sol_boot = []
    # iterate over bootstraps
    for i in range(n_bootstraps):
        sample = np.random.choice(
            np.arange(data.shape[0]), size=data.shape[0], replace=True
        )
        data_sample = data.iloc[sample]
        rho_sample = rho[sample]
        sol_boot.append(
            bulk_velocity_solver(
                data_sample,
                rho_sample,
                ra_col=ra_col,
                dec_col=dec_col,
                plx_col=plx_col,
                pmra_col=pmra_col,
                pmdec_col=pmdec_col,
                rv_col=rv_col,
                pmra_err_col=pmra_err_col,
                pmdec_err_col=pmdec_err_col,
                rv_err_col=rv_err_col,
                **kwargs
            )
        )

    return sol_boot
