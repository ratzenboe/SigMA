from scipy.optimize import minimize
import numpy as np
from numba import jit

# ----- GLOBAL CONSTANTS -----
# k: Astronomical unit expressed in km.yr/sec
# T: Transformation matrix for the coordinates (Equatorial to Galactic)
k = 4.740470446
T = np.array([[-0.0548755604, -0.8734370902, -0.4838350155],
              [+0.4941094279, -0.4448296300, +0.7469822445],
              [-0.8676661490, -0.1980763734, +0.4559837762]])
# ----------------------------


def prepare_inverse_transformation(ra, dec):
    # --------- Precalculate inverse matrices ----------------
    iT = np.linalg.inv(T)

    # Pre-calculating sine and cosine since it will be used multiple times
    cos_ra = np.cos(np.deg2rad(ra))
    cos_dec = np.cos(np.deg2rad(dec))
    sin_ra = np.sin(np.deg2rad(ra))
    sin_dec = np.sin(np.deg2rad(dec))

    # Calculating inverse of velocity conversion matrix
    iA_f = np.empty(shape=(ra.shape[0], 3, 3))
    for i, (cra, cdec, sra, sdec) in enumerate(zip(cos_ra, cos_dec, sin_ra, sin_dec)):
        buffer = np.array([[+cra * cdec, -sra, -cra * sdec],
                           [+sra * cdec, +cra, -sra * sdec],
                           [+sdec, 0, +cdec]])
        A = np.ndarray((3, 3), buffer=buffer)
        iA_f[i] = np.linalg.inv(A)

    return iT, iA_f


def transform_inverse(vel3d, iT, iA_f, plx, pmra, pmdec, rv, rv_err):
    # Calculate inverse matrix for each point on sky
    iV_f = np.empty(shape=(plx.shape[0], 3))
    for i, iA in enumerate(iA_f):
        iV_f[i] = iA @ iT @ vel3d
    # Transformed 3d velocities
    radial_velocity_calc = iV_f[:, 0]
    pmra_calc = (iV_f[:, 1] * plx) / k
    pmdec_calc = (iV_f[:, 2] * plx) / k

    # difference in radial velocity: might have a lot of nans -> remove them
    if np.count_nonzero(np.isnan(rv)) != plx.shape[0]:
        rv_isna = np.isnan(rv)
        rv_diff = (rv - radial_velocity_calc) ** 2 / rv_err ** 2
        rv_diff[rv_isna] = np.mean(rv_diff[~rv_isna])
    else:
        rv_diff = np.zeros_like(plx)
    # difference in pmra and dec
    pmra_diff = (pmra - pmra_calc) ** 2
    pmdec_diff = (pmdec - pmdec_calc) ** 2
    sum_diffs = pmra_diff + pmdec_diff + rv_diff
    sum_diffs = sum_diffs[~np.isnan(sum_diffs)]
    return np.mean(sum_diffs)  # , radial_velocity_calc


def transform_inverse_propermotions(vel3d, ra, dec, plx):
    iT, iA_f = prepare_inverse_transformation(ra, dec)
    # Calculate inverse matrix for each point on sky
    iV_f = np.empty(shape=(plx.shape[0], 3))
    for i, iA in enumerate(iA_f):
        iV_f[i] = iA @ iT @ vel3d
    # Transformed 3d velocities
    radial_velocity_calc = iV_f[:, 0]
    pmra_calc = (iV_f[:, 1] * plx) / k
    pmdec_calc = (iV_f[:, 2] * plx) / k
    return pmra_calc, pmdec_calc, radial_velocity_calc


def optimize_velocity(ra, dec, plx, pmra, pmdec, rv, rv_err, init_guess=np.zeros(3), do_minimize=True):
    """Find velocity that best describes the data"""
    iT, iA_f = prepare_inverse_transformation(ra, dec)
    if do_minimize:
        return minimize(fun=transform_inverse, x0=init_guess, args=(iT, iA_f, plx, pmra, pmdec, rv, rv_err))
    else:
        return transform_inverse(init_guess, iT, iA_f, plx, pmra, pmdec, rv, rv_err)


def prepare_transformation(ra, dec):
    """Transformation from ra,dec,plx,pmra,pmdec,rv -> u,v,w"""
    size = len(ra)
    # Values should be expressed in radians
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    # Pre-calculating sine and cosine since it will be used multiple times
    cos_ra = np.cos(ra)
    cos_dec = np.cos(dec)
    sin_ra = np.sin(ra)
    sin_dec = np.sin(dec)
    # ============ Calculating velocity ============
    # Coordinate matrix as in http://adsabs.harvard.edu/pdf/1987AJ.....93..864J7
    buffer = np.array([[[+ cos_ra * cos_dec], [- sin_ra], [- cos_ra * sin_dec]],
                       [[+ sin_ra * cos_dec], [+ cos_ra], [- sin_ra * sin_dec]],
                       [[+ sin_dec], np.zeros(size) if size == 1 else [np.zeros(size)], [+ cos_dec]]])
    A = np.ndarray((3, 3, size), buffer=buffer)
    return A


def transform_velocity(ra, dec, plx, pmra, pmdec, rv):
    # Get transfomation infos
    A = prepare_transformation(ra, dec)
    # Velocity components
    v1 = rv
    v2 = k * pmra / plx
    v3 = k * pmdec / plx
    V = np.array([v1, v2, v3]).T

    # Since the function arguments can be vectors, the variables A and V
    # may be 3D and 2D arrays. In order to calculate the velocities for
    # multiple inputs, the basic function was vectorized
    def space_velocity(a_i, v):
        # (T @ a).T = galactic coordinates
        return T @ a_i @ v

    func_vect = np.vectorize(space_velocity, signature='(n,n),(n)->(p)')
    velocities = func_vect(np.transpose(A, axes=(2, 0, 1)), V)
    return velocities


def prepare_transformation_single(ra, dec):
    # Values should be expressed in radians
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    # Pre-calculating sine and cosine since it will be used multiple times
    cos_ra = np.cos(ra)
    cos_dec = np.cos(dec)
    sin_ra = np.sin(ra)
    sin_dec = np.sin(dec)

    # ============ Calculating velocity ============
    # Coordinate matrix as in http://adsabs.harvard.edu/pdf/1987AJ.....93..864J7
    A = np.array([[+cos_ra * cos_dec, -sin_ra, -cos_ra * sin_dec],
                  [+sin_ra * cos_dec, +cos_ra, -sin_ra * sin_dec],
                  [+sin_dec, 0., +cos_dec]
                  ])
    return A


def transform_velocity_diff(rv, ra, dec, plx, pmra, pmdec, vec3d):
    # Get transfomation infos
    A = prepare_transformation_single(ra, dec)

    # Velocity components
    v1 = rv
    v2 = k * pmra / plx
    v3 = k * pmdec / plx
    V = np.hstack([v1, v2, v3])
    calc_vec3d = T @ A @ V
    difference = np.linalg.norm(vec3d - calc_vec3d)
    return difference


@jit(nopython=True)
def vr_solver(U, V, W, ra, dec, plx, pmra, pmdec):
    """Analytical minimizer for radial velocity with known
            1) 'optimal' velocity UVW
            2) sky positions ra, dec, plx
            3) proper motions pmra, pmdec
    """
    # -- important constants --
    k = 4.740470446
    # Individual elements of matrix T
    a, b, c = -0.0548755604, -0.8734370902, -0.4838350155
    d, e, f = +0.4941094279, -0.4448296300, +0.7469822445
    g, h, j = -0.8676661490, -0.1980763734, +0.4559837762
    # Transform ra/dec to radians
    dec = np.deg2rad(dec)
    ra = np.deg2rad(ra)
    # Compute optimal radial velocity that minimizes the L2 norm between UVW and
    # the velocity of the star at ra,dec,plx with pmra, pmdec
    numerator = (U*a*plx*np.cos(dec - ra)/2 + U*a*plx*np.cos(dec + ra)/2 -
                 U*b*plx*np.sin(dec - ra)/2 + U*b*plx*np.sin(dec + ra)/2 +
                 U*c*plx*np.sin(dec) +
                 V*d*plx*np.cos(dec - ra)/2 + V*d*plx*np.cos(dec + ra)/2 -
                 V*e*plx*np.sin(dec - ra)/2 + V*e*plx*np.sin(dec + ra)/2 +
                 V*f*plx*np.sin(dec) +
                 W*g*plx*np.cos(dec - ra)/2 + W*g*plx*np.cos(dec + ra)/2 -
                 W*h*plx*np.sin(dec - ra)/2 + W*h*plx*np.sin(dec + ra)/2 +
                 W*j*plx*np.sin(dec) +
                 a**2*k*pmdec*np.sin(2*dec)/4 + a**2*k*pmdec*np.sin(2*dec - 2*ra)/8 +
                 a**2*k*pmdec*np.sin(2*dec + 2*ra)/8 - a**2*k*pmra*np.sin(dec - 2*ra)/4 +
                 a**2*k*pmra*np.sin(dec + 2*ra)/4 + a*b*k*pmdec*np.cos(2*dec - 2*ra)/4 -
                 a*b*k*pmdec*np.cos(2*dec + 2*ra)/4 - a*b*k*pmra*np.cos(dec - 2*ra)/2 -
                 a*b*k*pmra*np.cos(dec + 2*ra)/2 - a*c*k*pmdec*np.cos(2*dec - ra)/2 -
                 a*c*k*pmdec*np.cos(2*dec + ra)/2 + a*c*k*pmra*np.cos(dec - ra)/2 -
                 a*c*k*pmra*np.cos(dec + ra)/2 +
                 b**2*k*pmdec*np.sin(2*dec)/4 - b**2*k*pmdec*np.sin(2*dec - 2*ra)/8 -
                 b**2*k*pmdec*np.sin(2*dec + 2*ra)/8 + b**2*k*pmra*np.sin(dec - 2*ra)/4 -
                 b**2*k*pmra*np.sin(dec + 2*ra)/4 + b*c*k*pmdec*np.sin(2*dec - ra)/2 -
                 b*c*k*pmdec*np.sin(2*dec + ra)/2 - b*c*k*pmra*np.sin(dec - ra)/2 -
                 b*c*k*pmra*np.sin(dec + ra)/2 -
                 c**2*k*pmdec*np.sin(2*dec)/2 +
                 d**2*k*pmdec*np.sin(2*dec)/4 + d**2*k*pmdec*np.sin(2*dec - 2*ra)/8 +
                 d**2*k*pmdec*np.sin(2*dec + 2*ra)/8 - d**2*k*pmra*np.sin(dec - 2*ra)/4 +
                 d**2*k*pmra*np.sin(dec + 2*ra)/4 + d*e*k*pmdec*np.cos(2*dec - 2*ra)/4 -
                 d*e*k*pmdec*np.cos(2*dec + 2*ra)/4 - d*e*k*pmra*np.cos(dec - 2*ra)/2 -
                 d*e*k*pmra*np.cos(dec + 2*ra)/2 - d*f*k*pmdec*np.cos(2*dec - ra)/2 -
                 d*f*k*pmdec*np.cos(2*dec + ra)/2 + d*f*k*pmra*np.cos(dec - ra)/2 -
                 d*f*k*pmra*np.cos(dec + ra)/2 +
                 e**2*k*pmdec*np.sin(2*dec)/4 - e**2*k*pmdec*np.sin(2*dec - 2*ra)/8 -
                 e**2*k*pmdec*np.sin(2*dec + 2*ra)/8 + e**2*k*pmra*np.sin(dec - 2*ra)/4 -
                 e**2*k*pmra*np.sin(dec + 2*ra)/4 + e*f*k*pmdec*np.sin(2*dec - ra)/2 -
                 e*f*k*pmdec*np.sin(2*dec + ra)/2 - e*f*k*pmra*np.sin(dec - ra)/2 -
                 e*f*k*pmra*np.sin(dec + ra)/2 -
                 f**2*k*pmdec*np.sin(2*dec)/2 +
                 g**2*k*pmdec*np.sin(2*dec)/4 + g**2*k*pmdec*np.sin(2*dec - 2*ra)/8 +
                 g**2*k*pmdec*np.sin(2*dec + 2*ra)/8 - g**2*k*pmra*np.sin(dec - 2*ra)/4 +
                 g**2*k*pmra*np.sin(dec + 2*ra)/4 + g*h*k*pmdec*np.cos(2*dec - 2*ra)/4 -
                 g*h*k*pmdec*np.cos(2*dec + 2*ra)/4 - g*h*k*pmra*np.cos(dec - 2*ra)/2 -
                 g*h*k*pmra*np.cos(dec + 2*ra)/2 - g*j*k*pmdec*np.cos(2*dec - ra)/2 -
                 g*j*k*pmdec*np.cos(2*dec + ra)/2 + g*j*k*pmra*np.cos(dec - ra)/2 -
                 g*j*k*pmra*np.cos(dec + ra)/2 +
                 h**2*k*pmdec*np.sin(2*dec)/4 - h**2*k*pmdec*np.sin(2*dec - 2*ra)/8 -
                 h**2*k*pmdec*np.sin(2*dec + 2*ra)/8 + h**2*k*pmra*np.sin(dec - 2*ra)/4 -
                 h**2*k*pmra*np.sin(dec + 2*ra)/4 + h*j*k*pmdec*np.sin(2*dec - ra)/2 -
                 h*j*k*pmdec*np.sin(2*dec + ra)/2 - h*j*k*pmra*np.sin(dec - ra)/2 -
                 h*j*k*pmra*np.sin(dec + ra)/2 -
                 j**2*k*pmdec*np.sin(2*dec)/2)
    denominator = plx*(a**2*np.cos(dec)**2*np.cos(ra)**2 +
                        2*a*b*np.sin(ra)*np.cos(dec)**2*np.cos(ra) +
                        a*c*(np.sin(2*dec - ra) + np.sin(2*dec + ra))/2 +
                        b**2*np.sin(ra)**2*np.cos(dec)**2 +
                        b*c*(np.cos(2*dec - ra) - np.cos(2*dec + ra))/2 +
                        c**2*np.sin(dec)**2 + d**2*np.cos(dec)**2*np.cos(ra)**2 +
                        2*d*e*np.sin(ra)*np.cos(dec)**2*np.cos(ra) +
                        d*f*(np.sin(2*dec - ra) + np.sin(2*dec + ra))/2 +
                        e**2*np.sin(ra)**2*np.cos(dec)**2 +
                        e*f*(np.cos(2*dec - ra) - np.cos(2*dec + ra))/2 +
                        f**2*np.sin(dec)**2 + g**2*np.cos(dec)**2*np.cos(ra)**2 +
                        2*g*h*np.sin(ra)*np.cos(dec)**2*np.cos(ra) +
                        g*j*(np.sin(2*dec - ra) + np.sin(2*dec + ra))/2 +
                        h**2*np.sin(ra)**2*np.cos(dec)**2 +
                        h*j*(np.cos(2*dec - ra) - np.cos(2*dec + ra))/2 + j**2*np.sin(dec)**2
                       )
    vr_minimizer = numerator/denominator
    return vr_minimizer
