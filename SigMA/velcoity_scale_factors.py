import numpy as np
from pynverse import inversefunc
from scipy.special import erf


def cdf_sum_of_gaussians(y, sigma, y0, y1):
    cdf = (np.sqrt(2/np.pi) * sigma *
           (np.exp(-(y0 - y)**2 / (2 * sigma**2)) - np.exp(-(y1 - y)**2 / (2 * sigma**2))) +
           (y - y0)*erf((y - y0)/(np.sqrt(2)*sigma)) + (y1 - y)*erf((y - y1)/(np.sqrt(2)*sigma))
           )
    cdf /= (y1 - y0)*2
    cdf += 0.5
    return cdf


def linear_fit_f(min_dist, max_dist):
    """Best linear fit to scaling function over distance to Gagne+2018 and Cantat-Gaudin+2020 data
    :returns: y-positions of best fit line given the min and max distance of an object
              needed for computing scaling parameters for that region
    """
    # --- data deviation around best fitting line
    sigma = 4.5369
    # --- parameters of best fitting line
    intercept, slope = 3.521, 0.0372
    y0 = intercept + slope*min_dist
    y1 = intercept + slope*max_dist
    return sigma, y0, y1


def scale_samples(min_dist, max_dist, n_samples=10):
    # --- Parameters for sample distribution
    sigma, y0, y1 = linear_fit_f(min_dist, max_dist)
    # --- draw n samples from distribution at equally sized intervals
    samples_quantile = np.linspace(0.05, 0.95, n_samples)
    invcdf = inversefunc(cdf_sum_of_gaussians, args=(sigma, y0, y1))
    return invcdf(samples_quantile)

