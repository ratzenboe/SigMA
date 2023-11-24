import numpy as np
from scipy.special import erfc


# Gagne's helper functions
ra_pol = 192.8595
dec_pol = 27.12825
# Initiate some secondary variables
sin_dec_pol = np.sin(np.radians(dec_pol))
cos_dec_pol = np.cos(np.radians(dec_pol))

# J2000.0 Galactic latitude gb of the Celestial North pole (dec=90 degrees) from Carrol and Ostlie
l_north = 122.932

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

# A very small number used for numerical stability
tiny_number = 1e-318


def equatorial_galactic(ra, dec):
    """Transforms equatorial coordinates (ra,dec) to Galactic coordinates (gl,gb). All inputs must be numpy arrays of the same dimension

    param ra: Right ascension (degrees)
    param dec: Declination (degrees)
    output (gl,gb): Tuple containing Galactic longitude and latitude (degrees)
    """
    # Check for parameter consistency
    num_stars = np.size(ra)
    if np.size(dec) != num_stars:
        raise ValueError(
            "The dimensions ra and dec do not agree. They must all be numpy arrays of the same length."
        )
    # Compute intermediate quantities
    ra_m_ra_pol = ra - ra_pol
    sin_ra = np.sin(np.radians(ra_m_ra_pol))
    cos_ra = np.cos(np.radians(ra_m_ra_pol))
    sin_dec = np.sin(np.radians(dec))
    cos_dec = np.cos(np.radians(dec))
    # Compute Galactic latitude
    gamma = sin_dec_pol * sin_dec + cos_dec_pol * cos_dec * cos_ra
    gb = np.degrees(np.arcsin(gamma))
    # Compute Galactic longitude
    x1 = cos_dec * sin_ra
    x2 = (sin_dec - sin_dec_pol * gamma) / cos_dec_pol
    gl = l_north - np.degrees(np.arctan2(x1, x2))
    # gl = (gl+360.) % 360.
    gl = np.mod(gl, 360.0)  # might be better
    # Return Galactic coordinates tuple
    return gl, gb


def equatorial_UVW(
    ra,
    dec,
    pmra,
    pmdec,
    rv,
    dist,
    pmra_error=None,
    pmdec_error=None,
    rv_error=None,
    dist_error=None,
):
    """
    Transforms equatorial coordinates (ra,dec), proper motion (pmra,pmdec), radial velocity and distance to space velocities UVW. All inputs must be numpy arrays of the same dimension.

    param ra: Right ascension (degrees)
    param dec: Declination (degrees)
    param pmra: Proper motion in right ascension (milliarcsecond per year). 	Must include the cos(delta) term
    param pmdec: Proper motion in declination (milliarcsecond per year)
    param rv: Radial velocity (kilometers per second)
    param dist: Distance (parsec)
    param ra_error: Error on right ascension (degrees)
    param dec_error: Error on declination (degrees)
    param pmra_error: Error on proper motion in right ascension (milliarcsecond per year)
    param pmdec_error: Error on proper motion in declination (milliarcsecond per year)
    param rv_error: Error on radial velocity (kilometers per second)
    param dist_error: Error on distance (parsec)

    output (U,V,W): Tuple containing Space velocities UVW (kilometers per second)
    output (U,V,W,EU,EV,EW): Tuple containing Space velocities UVW and their measurement errors, used if any measurement errors are given as inputs (kilometers per second)
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

    # Calculate UVW
    reduced_dist = kappa * dist
    U = T1 * rv + T2 * pmra * reduced_dist + T3 * pmdec * reduced_dist
    V = T4 * rv + T5 * pmra * reduced_dist + T6 * pmdec * reduced_dist
    W = T7 * rv + T8 * pmra * reduced_dist + T9 * pmdec * reduced_dist

    # Return only (U, V, W) tuple if no errors are set
    if (
        pmra_error is None
        and pmdec_error is None
        and rv_error is None
        and dist_error is None
    ):
        return (U, V, W)

    # Propagate errors if they are specified
    reduced_dist_error = kappa * dist_error

    # Calculate derivatives
    T23_pm = np.sqrt((T2 * pmra) ** 2 + (T3 * pmdec) ** 2)
    T23_pm_error = np.sqrt((T2 * pmra_error) ** 2 + (T3 * pmdec_error) ** 2)
    EU_rv = T1 * rv_error
    EU_pm = T23_pm_error * reduced_dist
    EU_dist = T23_pm * reduced_dist_error
    EU_dist_pm = T23_pm_error * reduced_dist_error

    T56_pm = np.sqrt((T5 * pmra) ** 2 + (T6 * pmdec) ** 2)
    T56_pm_error = np.sqrt((T5 * pmra_error) ** 2 + (T6 * pmdec_error) ** 2)
    EV_rv = T4 * rv_error
    EV_pm = T56_pm_error * reduced_dist
    EV_dist = T56_pm * reduced_dist_error
    EV_dist_pm = T56_pm_error * reduced_dist_error

    T89_pm = np.sqrt((T8 * pmra) ** 2 + (T9 * pmdec) ** 2)
    T89_pm_error = np.sqrt((T8 * pmra_error) ** 2 + (T9 * pmdec_error) ** 2)
    EW_rv = T7 * rv_error
    EW_pm = T89_pm_error * reduced_dist
    EW_dist = T89_pm * reduced_dist_error
    EW_dist_pm = T89_pm_error * reduced_dist_error

    # Calculate error bars
    EU = np.sqrt(EU_rv**2 + EU_pm**2 + EU_dist**2 + EU_dist_pm**2)
    EV = np.sqrt(EV_rv**2 + EV_pm**2 + EV_dist**2 + EV_dist_pm**2)
    EW = np.sqrt(EW_rv**2 + EW_pm**2 + EW_dist**2 + EW_dist_pm**2)

    # Return measurements and error bars
    return U, V, W, EU, EV, EW


def parabolic_cylinder_f5_mod(x):
    """
    Calculates the real part of the "modified" Parabolic Cylinder Function D of index v=-5.

    The regular function D(-5,x) is equivalent to the real part of:
        from scipy.special import pbdv
        return pbdv(-5,x)

    And is equivalent to the mathematical expression:
        exp(x^2/4)/24 * (sqrt(pi/2)*(x^4+6*x^2+3)*erfc(x/sqrt(2)) - exp(-x^2/2)*(x^3+5*x))

    The modified parabolic cylinder does away with the exp(x^2/4) term to improve numerical stability, and instead returns:
        (sqrt(pi/2)*(x^4+6*x^2+3)*erfc(x/sqrt(2)) - exp(-x^2/2)*(x^3+5*x))/24

    """

    # Define shortcuts for efficiency
    sqrt2 = np.sqrt(2.0)
    sqrt_halfpi = np.sqrt(np.pi) / sqrt2
    x_over_sqrt2 = x / sqrt2
    erfc_x_over_sqrt2 = erfc(x_over_sqrt2)
    epsilon = np.exp(-(x**2) / 2.0)

    # Calculate the output
    y = (
        1
        / 24.0
        * (
            sqrt_halfpi * (x**4 + 6.0 * x**2 + 3.0) * erfc_x_over_sqrt2
            - epsilon * (x**3 + 5.0 * x)
        )
    )

    return y


def equatorial_XYZ(ra, dec, dist, dist_error=None):
    """
    Transforms equatorial coordinates (ra,dec) and distance to Galactic position XYZ. All inputs must be numpy arrays of the same dimension.

    param ra: Right ascension (degrees)
    param dec: Declination (degrees)
    param dist: Distance (parsec)
    param dist_error: Error on distance (parsec)

    output (X,Y,Z): Tuple containing Galactic position XYZ (parsec)
    output (X,Y,Z,EX,EY,EZ): Tuple containing Galactic position XYZ and their measurement errors, used if any measurement errors are given as inputs (parsec)
    """
    # Compute Galactic coordinates
    gl, gb = equatorial_galactic(ra, dec)
    cos_gl = np.cos(np.radians(gl))
    cos_gb = np.cos(np.radians(gb))
    sin_gl = np.sin(np.radians(gl))
    sin_gb = np.sin(np.radians(gb))

    X = cos_gb * cos_gl * dist
    Y = cos_gb * sin_gl * dist
    Z = sin_gb * dist
    if dist_error is None:
        return X, Y, Z

    X_dist = cos_gb * cos_gl
    EX = np.abs(X_dist * dist_error)
    Y_dist = cos_gb * sin_gl
    EY = np.abs(Y_dist * dist_error)
    Z_dist = sin_gb
    EZ = np.abs(Z_dist * dist_error)

    return X, Y, Z, EX, EY, EZ
