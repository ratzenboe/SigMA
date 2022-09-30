import numpy as np
from NoiseRemoval.OptimalVelocity import prepare_inverse_transformation

# ----- GLOBAL CONSTANTS -----
# k: Astronomical unit expressed in km.yr/sec
# T: Transformation matrix for the coordinates (Equatorial to Galactic)
k = 4.740470446
T = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
              [+0.4941094279, -0.4448296300, +0.7469822445],
              [-0.8676661490, -0.1980763734, +0.4559837762]])
# ----------------------------


def transform_inverse_XYZ(coords_gal):
    """Usage:
    coords_gal = data[['x','y','z']].values
    """
    # Inverse of T
    iT = np.linalg.inv(T)
    # Reversing coordinates to ICRS frame
    coords_icrs = iT @ coords_gal.T
    # Calculating ra, dec and parallax from x, y, z
    x, y, z = coords_icrs
    parallax = 1000 / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ra = np.arctan2(y, x)
    dec = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    # Not to return negative degrees
    # Apparently ra is all positive but dec does not follow the pattern
    vals_lt_zero = np.rad2deg(ra) < 0
    ra[vals_lt_zero] = 360 + np.rad2deg(ra[vals_lt_zero])
    ra[~vals_lt_zero] = np.rad2deg(ra[~vals_lt_zero])
    return np.vstack([ra, np.rad2deg(dec), parallax]).T


def transform_inverse_UVW(vel3d, ra, dec, plx):
    iT, iA_f = prepare_inverse_transformation(ra, dec)
    # Calculate inverse matrix for each point on sky
    iV_f = np.empty(shape=(plx.shape[0], 3))
    for i, iA in enumerate(iA_f):
        iV_f[i] = iA @ iT @ vel3d[i]
    # Transformed 3d velocities
    radial_velocity_calc = iV_f[:, 0]
    pmra_calc = (iV_f[:, 1] * plx) / k
    pmdec_calc = (iV_f[:, 2] * plx) / k
    return np.vstack([pmra_calc, pmdec_calc, radial_velocity_calc]).T



