import numpy as np


# Galactic Coordinates matrix
TGAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ]
)
# Initiate some global constants
# 1 AU/yr to km/s divided by 1000
kappa = 0.004743717361


def transform_(
        ra,
        dec,
        plx,
        C
):
    """
    Transforms the observed covariance matrix of on sky velocities (pmra, pmdec, radial_velocity) into the covariance matrix of space velocities (U,V,W). All inputs must be numpy arrays of the same dimension.

    param ra: Right ascension (degrees)
    param dec: Declination (degrees)
    param plx: Parallax (mas)
    param C: Covariance matrix of pmra, pmdec, radial_velocity (mas/yr, mas/yr, km/s)

    output C_uvw: Covariance matrix in the new Galactic coordinate system (U,V,W [km/s])
    """
    # Compute elements of the T matrix
    cos_ra = np.cos(np.radians(ra))
    cos_dec = np.cos(np.radians(dec))
    sin_ra = np.sin(np.radians(ra))
    sin_dec = np.sin(np.radians(dec))
    T1 = (
            TGAL[0, 0] * cos_ra * cos_dec
            + TGAL[0, 1] * sin_ra * cos_dec
            + TGAL[0, 2] * sin_dec
    )
    T2 = -TGAL[0, 0] * sin_ra + TGAL[0, 1] * cos_ra
    T3 = (
            -TGAL[0, 0] * cos_ra * sin_dec
            - TGAL[0, 1] * sin_ra * sin_dec
            + TGAL[0, 2] * cos_dec
    )
    T4 = (
            TGAL[1, 0] * cos_ra * cos_dec
            + TGAL[1, 1] * sin_ra * cos_dec
            + TGAL[1, 2] * sin_dec
    )
    T5 = -TGAL[1, 0] * sin_ra + TGAL[1, 1] * cos_ra
    T6 = (
            -TGAL[1, 0] * cos_ra * sin_dec
            - TGAL[1, 1] * sin_ra * sin_dec
            + TGAL[1, 2] * cos_dec
    )
    T7 = (
            TGAL[2, 0] * cos_ra * cos_dec
            + TGAL[2, 1] * sin_ra * cos_dec
            + TGAL[2, 2] * sin_dec
    )
    T8 = -TGAL[2, 0] * sin_ra + TGAL[2, 1] * cos_ra
    T9 = (
            -TGAL[2, 0] * cos_ra * sin_dec
            - TGAL[2, 1] * sin_ra * sin_dec
            + TGAL[2, 2] * cos_dec
    )

    # Calculate jacobi matrix
    reduced_dist = kappa * (1000 / plx)

    jacobian = np.zeros((ra.shape[0], 3, 3))
    jacobian[:, 0, 0] = T2 * reduced_dist
    jacobian[:, 0, 1] = T3 * reduced_dist
    jacobian[:, 0, 2] = T1
    jacobian[:, 1, 0] = T5 * reduced_dist
    jacobian[:, 1, 1] = T6 * reduced_dist
    jacobian[:, 1, 2] = T4
    jacobian[:, 2, 0] = T8 * reduced_dist
    jacobian[:, 2, 1] = T9 * reduced_dist
    jacobian[:, 2, 2] = T7

    # Calculate derivatives
    C_uvw = np.einsum('ijk, ikl, ilm -> ijm', jacobian, C, np.transpose(jacobian, (0, 2, 1)))
    return C_uvw
