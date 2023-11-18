import numpy as np
from NoiseRemoval.gagne_helper_functions import (
    equatorial_galactic,
    equatorial_UVW,
    equatorial_XYZ,
    parabolic_cylinder_f5_mod,
)


# Galactic Coordinates matrix
TGAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ]
)
# 1 AU/yr to km/s divided by 1000
kappa = 0.004743717361

# A very small number used for numerical stability
tiny_number = 1e-318


# Implement Gagne's procedure
def compute_Gagne_vectorized(
    ra,
    dec,
    parallax,
    pmra,
    pmdec,
    rv,
    # Errors
    pmra_error,
    pmdec_error,
    parallax_error,
    rv_error,
    # Gauss parameters
    precision_matrix,
    center_vec,
):
    """
    param precision_matrix: Inverse of the covariance matrix [XYZUVW] of the multivariate Gaussian model (mixed units of pc and km/s)
    param center_vec: Central XYZUVW position of the multivariate Gaussian model (mixed units of pc and km/s)
    """
    # pre-process distances
    dist = 1000 / parallax
    dist_error = 1000 / parallax**2 * parallax_error

    num_stars = np.size(ra)
    # Compute Galactic coordinates
    gl, gb = equatorial_galactic(ra, dec)

    # lambda is defined in Gagne et al. (2017, ApJS, X, Y, equation 7)
    cos_gl = np.cos(np.radians(gl))
    cos_gb = np.cos(np.radians(gb))
    sin_gl = np.sin(np.radians(gl))
    sin_gb = np.sin(np.radians(gb))
    lambda_vector = np.array([cos_gb * cos_gl, cos_gb * sin_gl, sin_gb]).transpose()

    # Build matrices A and B to convert sky quantities in the Galactic coordinates frame. The A matrix is defined in Gagne et al. (2017, ApJS, X, Y, equation 7)
    A_matrix = np.zeros((num_stars, 3, 3))
    cos_ra = np.cos(np.radians(ra))
    cos_dec = np.cos(np.radians(dec))
    sin_ra = np.sin(np.radians(ra))
    sin_dec = np.sin(np.radians(dec))
    A_matrix[:, 0, 0] = cos_ra * cos_dec
    A_matrix[:, 1, 0] = sin_ra * cos_dec
    A_matrix[:, 2, 0] = sin_dec
    A_matrix[:, 0, 1] = -sin_ra
    A_matrix[:, 1, 1] = cos_ra
    A_matrix[:, 0, 2] = -cos_ra * sin_dec
    A_matrix[:, 1, 2] = -sin_ra * sin_dec
    A_matrix[:, 2, 2] = cos_dec

    # The B matrix is not directly referenced in the BANYAN Sigma paper.
    B_matrix = np.tensordot(TGAL, A_matrix, axes=(1, 1)).swapaxes(
        0, 1
    )  # Own code to speed up things

    # The M vector is defined in Gagne et al. (2017, ApJS, X, Y, equation 7)
    M_vector = np.einsum(
        "ijk,k->ij", B_matrix, np.array([1.0, 0.0, 0.0])
    )  # Own code to speed up things

    # The N vector is defined in Gagne et al. (2017, ApJS, X, Y, equation 7)
    N_vector_sub = np.array(
        [np.zeros(num_stars), np.array(kappa * pmra), np.array(kappa * pmdec)]
    ).transpose()
    N_vector = np.einsum("ijk,ik->ij", B_matrix, N_vector_sub)

    # OMEGA is defined in Gagne et al. (2017, ApJS, X, Y, equation 6)
    zero_vector = np.zeros([num_stars, 3])
    OMEGA_vector = np.concatenate((zero_vector, M_vector), axis=1)

    # GAMMA is defined in Gagne et al. (2017, ApJS, X, Y, equation 6)
    GAMMA_vector = np.concatenate((lambda_vector, N_vector), axis=1)

    # tau is defined in Gagne et al. (2017, ApJS, X, Y, equation 5)
    TAU_vector = np.repeat(center_vec.reshape(1, 6), num_stars, axis=0)

    # Take scalar products in multivariate space
    OMEGA_OMEGA = np.einsum("ni,nj,ij->n", OMEGA_vector, OMEGA_vector, precision_matrix)
    GAMMA_GAMMA = np.einsum("ni,nj,ij->n", GAMMA_vector, GAMMA_vector, precision_matrix)
    OMEGA_GAMMA = np.einsum("ni,nj,ij->n", OMEGA_vector, GAMMA_vector, precision_matrix)
    OMEGA_TAU = np.einsum("ni,nj,ij->n", OMEGA_vector, TAU_vector, precision_matrix)
    GAMMA_TAU = np.einsum("ni,nj,ij->n", GAMMA_vector, TAU_vector, precision_matrix)
    TAU_TAU = np.einsum("ni,nj,ij->n", TAU_vector, TAU_vector, precision_matrix)

    # Propagate radial velocity and distance measurements to relevant scalar products
    # ---- Distances -----
    norm = np.maximum(dist_error, 1e-3) ** 2
    GAMMA_GAMMA += 1.0 / norm
    GAMMA_TAU += dist / norm
    TAU_TAU += dist**2 / norm
    # ---- Radial velocities -----
    # Find where measured RVs are finite
    finite_ind = np.where(np.isfinite(rv) & np.isfinite(rv_error))
    if np.size(finite_ind) != 0:
        norm = np.maximum(rv_error[finite_ind], 1e-3) ** 2
        OMEGA_OMEGA[finite_ind] += 1.0 / norm
        OMEGA_TAU[finite_ind] += rv[finite_ind] / norm
        TAU_TAU[finite_ind] += rv[finite_ind] ** 2 / norm

    # Calculate the determinant of the precision matrix unless it is given as a parameter
    precision_matrix_determinant = np.linalg.det(precision_matrix)
    if precision_matrix_determinant <= 0:
        raise ValueError(
            "The determinant of the precision matrix bust be positive and non-zero !"
        )

    # Calculate optimal distance and radial velocity
    beta = (GAMMA_GAMMA - OMEGA_GAMMA**2 / OMEGA_OMEGA) / 2.0
    if np.nanmin(beta) < 0:
        raise ValueError("beta has an ill-defined value !")
    gamma = OMEGA_GAMMA * OMEGA_TAU / OMEGA_OMEGA - GAMMA_TAU
    dist_optimal = (np.sqrt(gamma**2 + 32.0 * beta) - gamma) / (4.0 * beta)
    rv_optimal = (4.0 - GAMMA_GAMMA * dist_optimal**2 + GAMMA_TAU * dist_optimal) / (
        OMEGA_GAMMA * dist_optimal
    )

    # Create arrays that contain the measured RV and distance if available, or the optimal values otherwise
    # ---- Distances (this currently just uses the input distances) -----
    dist_optimal_or_measured = dist_optimal
    finite_ind = np.where(np.isfinite(dist) & np.isfinite(dist_error))
    if np.size(finite_ind) != 0:
        dist_optimal_or_measured[finite_ind] = dist[finite_ind]
    # ---- Radial velocities -----
    rv_optimal_or_measured = rv_optimal
    finite_ind = np.where(np.isfinite(rv) & np.isfinite(rv_error))
    if np.size(finite_ind) != 0:
        rv_optimal_or_measured[finite_ind] = rv[finite_ind]

    # Propagate proper motion measurement errors
    EX = np.zeros(num_stars)
    EY = np.zeros(num_stars)
    EZ = np.zeros(num_stars)
    U, V, W, EU, EV, EW = equatorial_UVW(
        ra,
        dec,
        pmra,
        pmdec,
        rv_optimal_or_measured,
        dist_optimal_or_measured,
        pmra_error=pmra_error,
        pmdec_error=pmdec_error,
        rv_error=rv_error,
        dist_error=dist_error,
    )

    # Determine by how much the diagonal of the covariance matrix must be inflated to account for the measurement errors
    covariance_matrix = np.linalg.inv(precision_matrix)
    covariance_diagonal = np.diag(covariance_matrix)
    inflation_array = np.array([EX, EY, EZ, EU, EV, EW]).transpose()
    inflation_factors = 1.0 + inflation_array**2 / np.repeat(
        covariance_diagonal.reshape(1, 6), num_stars, axis=0
    )

    # Calculate how much the determinant of the covariance matrices must be inflated
    inflation_covariance_determinant = np.exp(np.sum(np.log(inflation_factors), axis=1))

    # Make sure that no matrix becomes unphysical
    if np.nanmin(inflation_covariance_determinant) <= 0:
        raise ValueError(
            "At least one covariance matrix has a negative or null determinant as a consequence of the measurement errors !"
        )

    # Calculate new determinants for the precision matrices
    precision_matrix_inflated_determinant = (
        precision_matrix_determinant / inflation_covariance_determinant
    )

    # Apply this to the precision matrices (again own implementation to speed up things)
    precision_matrix_inflated = np.einsum(
        "ni,ij,nj->nij",
        1.0 / np.sqrt(inflation_factors),
        precision_matrix,
        1.0 / np.sqrt(inflation_factors),
    )

    # Recalculate the scalar products with new precision matrices (again own implementation to speed up things)
    OMEGA_OMEGA = np.einsum(
        "ni,nij,nj->n", OMEGA_vector, precision_matrix_inflated, OMEGA_vector
    )
    GAMMA_GAMMA = np.einsum(
        "ni,nij,nj->n", GAMMA_vector, precision_matrix_inflated, GAMMA_vector
    )
    OMEGA_GAMMA = np.einsum(
        "ni,nij,nj->n", OMEGA_vector, precision_matrix_inflated, GAMMA_vector
    )
    OMEGA_TAU = np.einsum(
        "ni,nij,nj->n", OMEGA_vector, precision_matrix_inflated, TAU_vector
    )
    GAMMA_TAU = np.einsum(
        "ni,nij,nj->n", GAMMA_vector, precision_matrix_inflated, TAU_vector
    )
    TAU_TAU = np.einsum(
        "ni,nij,nj->n", TAU_vector, precision_matrix_inflated, TAU_vector
    )

    # If radial velocity or distance measurements are given, propagate them to the relevant scalar products
    # ---- Distances -----
    norm = np.maximum(dist_error, 1e-3) ** 2
    GAMMA_GAMMA += 1.0 / norm
    GAMMA_TAU += dist / norm
    TAU_TAU += dist**2 / norm
    # ---- Radial velocities -----
    # Find where measured RVs are finite
    finite_ind = np.where(np.isfinite(rv) & np.isfinite(rv_error))
    if np.size(finite_ind) != 0:
        norm = np.maximum(rv_error[finite_ind], 1e-3) ** 2
        OMEGA_OMEGA[finite_ind] += 1.0 / norm
        OMEGA_TAU[finite_ind] += rv[finite_ind] / norm
        TAU_TAU[finite_ind] += rv[finite_ind] ** 2 / norm

    # Update optimal distance and radial velocity
    beta = (GAMMA_GAMMA - OMEGA_GAMMA**2 / OMEGA_OMEGA) / 2.0
    if np.nanmin(beta) < 0:
        raise ValueError("beta has an ill-defined value !")
    gamma = OMEGA_GAMMA * OMEGA_TAU / OMEGA_OMEGA - GAMMA_TAU
    dist_optimal = (np.sqrt(gamma**2 + 32.0 * beta) - gamma) / (4.0 * beta)
    rv_optimal = (4.0 - GAMMA_GAMMA * dist_optimal**2 + GAMMA_TAU * dist_optimal) / (
        OMEGA_GAMMA * dist_optimal
    )

    # Calculate error bars on the optimal distance and radial velocity
    edist_optimal = 1.0 / np.sqrt(GAMMA_GAMMA)
    erv_optimal = 1.0 / np.sqrt(OMEGA_OMEGA)

    # Calculate final quantities for ln probability
    zeta = (TAU_TAU - OMEGA_TAU**2 / OMEGA_OMEGA) / 2.0
    xarg = gamma / np.sqrt(2.0 * beta)

    lnP_coeff = (
        -0.5 * np.log(OMEGA_OMEGA)
        - 2.5 * np.log(beta)
        + 0.5 * np.log(precision_matrix_inflated_determinant)
    )
    lnP_part1 = xarg**2 / 2.0 - zeta
    lnP_part2 = np.log(np.maximum(parabolic_cylinder_f5_mod(xarg), tiny_number))
    lnP = lnP_coeff + lnP_part1 + lnP_part2

    # Create arrays that contain the measured RV and distance if available, or the optimal values otherwise
    # ---- Distances (this currently just uses the input distances) -----
    dist_optimal_or_measured = dist_optimal
    edist_optimal_or_measured = edist_optimal
    dist_optimal_or_measured[finite_ind] = dist[finite_ind]
    edist_optimal_or_measured[finite_ind] = dist_error[finite_ind]
    # ---- Radial velocities -----
    rv_optimal_or_measured = rv_optimal
    erv_optimal_or_measured = erv_optimal
    finite_ind = np.where(np.isfinite(rv) & np.isfinite(rv_error))
    if np.size(finite_ind) != 0:
        rv_optimal_or_measured[finite_ind] = rv[finite_ind]
        erv_optimal_or_measured[finite_ind] = rv_error[finite_ind]

    # Calculate XYZ and UVW positions at the optimal (or measured) RV and distance
    X, Y, Z, EX, EY, EZ = equatorial_XYZ(
        ra, dec, dist_optimal_or_measured, dist_error=edist_optimal_or_measured
    )
    U, V, W, EU, EV, EW = equatorial_UVW(
        ra,
        dec,
        pmra,
        pmdec,
        rv_optimal_or_measured,
        dist_optimal_or_measured,
        pmra_error=pmra_error,
        pmdec_error=pmdec_error,
        rv_error=erv_optimal_or_measured,
        dist_error=edist_optimal_or_measured,
    )
    XYZUVW = np.array([X, Y, Z, U, V, W]).transpose()
    EXYZUVW = np.array([EX, EY, EZ, EU, EV, EW]).transpose()

    # Calculate the Mahalanobis distance from the optimal position to the Gaussian model
    vec = XYZUVW - TAU_vector
    mahalanobis = np.sqrt(
        np.einsum("ni,nij,nj->n", vec, precision_matrix_inflated, vec)
    )

    # Calculate the XYZ (pc) and UVW (km/s) separations from the optimal position to the center of the Gaussian model
    XYZ_sep = np.sqrt(np.sum((XYZUVW[:, 0:3] - TAU_vector[:, 0:3]) ** 2, axis=1))
    UVW_sep = np.sqrt(np.sum((XYZUVW[:, 3:6] - TAU_vector[:, 3:6]) ** 2, axis=1))

    # Calculate the 3D N-sigma distances from the optimal position to the center of the Gaussian models
    XYZ_sig = np.sqrt(
        np.einsum(
            "ni,nij,nj->n",
            vec[:, 0:3],
            precision_matrix_inflated[:, 0:3, 0:3],
            vec[:, 0:3],
        )
    )
    UVW_sig = np.sqrt(
        np.einsum(
            "ni,nij,nj->n",
            vec[:, 3:6],
            precision_matrix_inflated[:, 3:6, 3:6],
            vec[:, 3:6],
        )
    )

    return rv_optimal_or_measured, XYZUVW, XYZ_sig, UVW_sig, mahalanobis
