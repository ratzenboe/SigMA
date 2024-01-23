import numpy as np
from NoiseRemoval.OptimalVelocity import prepare_inverse_transformation
from NoiseRemoval.bulk_velocity_solver import dense_sample
from scipy import optimize, stats


def ll_matrix(
        theta, iT, iA_f, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err
):
    # Cluster dispersion important for updated log likelihood calculation
    v1, v2, v3, cluster_disp = theta
    vel_3d = np.array([v1, v2, v3])
    # Calculate inverse matrix for each point on sky
    iV_f = np.empty(shape=(plx.shape[0], 3))
    for i, iA in enumerate(iA_f):
        iV_f[i] = iA @ iT @ vel_3d
    # Transformed observed velocities
    rv_calc = iV_f[:, 0]
    pmra_calc = (iV_f[:, 1] * plx) / 4.740470446
    pmdec_calc = (iV_f[:, 2] * plx) / 4.740470446

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


def optimize_velocity_matrix(
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
    iT, iA_f = prepare_inverse_transformation(ra, dec)

    sol = optimize.minimize(
        # fun=transform_inverse,
        fun=ll_matrix,
        x0=init_guess,
        method=method,
        args=(iT, iA_f, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err),
        tol=1e-02,
        **kwargs
    )
    return sol


def bulk_velocity_solver_matrix(
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
    sol = optimize_velocity_matrix(
        ra, dec, plx, pmra, pmdec, rv, pmra_err, pmdec_err, rv_err, **kwargs
    )
    return sol


def bootstrap_bulk_velocity_solver_matrix(
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
            bulk_velocity_solver_matrix(
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
