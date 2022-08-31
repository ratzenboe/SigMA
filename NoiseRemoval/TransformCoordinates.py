import numpy as np


def transform_inverse(coords_gal, vels):
    """
    coord = data[['x','y','z']].values
    vel = data[['Vx','Vy','Vz']].values
    ra, dec, parallax, pmra, pmdec, radial_velocity = transform_inverse(coord, vel)
    """
    k = 4.740470446

    # Inverse of T
    T = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
                  [+0.4941094279, -0.4448296300, +0.7469822445],
                  [-0.8676661490, -0.1980763734, +0.4559837762]])
    iT = np.linalg.inv(T)

    # Reversing coordinates to ICRS frame
    coords_icrs = iT @ coords_gal

    # Calculating ra, dec and parallax from x, y, z
    x, y, z = coords_icrs
    parallax = 1 / np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ra = np.arctan2(y, x)
    dec = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))

    # Pre-calculating sine and cosine since it will be used multiple times
    cos_ra = np.cos(ra)
    cos_dec = np.cos(dec)
    sin_ra = np.sin(ra)
    sin_dec = np.sin(dec)

    # Calculating inverse of velocity conversion matrix
    buffer = np.array([[+ cos_ra * cos_dec, - sin_ra, - cos_ra * sin_dec],
                       [+ sin_ra * cos_dec, + cos_ra, - sin_ra * sin_dec],
                       [+ sin_dec, 0, + cos_dec]])
    A = np.ndarray((3, 3), buffer=buffer)
    iA = np.linalg.inv(A)

    iV = iA @ iT @ vels
    radial_velocity = iV[0]
    pmra = (iV[1] * parallax) / k
    pmdec = (iV[2] * parallax) / k

    # Not to return negative degrees
    # Apparently ra is all positive but dec does not follow the pattern
    ra = 360 + np.rad2deg(ra) if np.rad2deg(ra) < 0 else np.rad2deg(ra)

    return [ra, np.rad2deg(dec), parallax, pmra, pmdec, radial_velocity]


