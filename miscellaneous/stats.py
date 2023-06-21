import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def running_median(X, Y, total_bins=20, range_calc_median=None):
    """
    Calculate the median over a scatter plot: Median of y-values binned on x-axis
    :param X: array of values for x axis
    :param Y: array of values for y axis
    :param total_bins: number of bins for median calculation (on x-axis)
    :param range_calc_median: x-axis range in which the running median should be computed
    """
    if range_calc_median is None:
        xmin, xmax = X.min(), X.max()
    else:
        xmin, xmax = range_calc_median
    # get bins
    bins = np.linspace(xmin, xmax, total_bins)
    delta = bins[1]-bins[0]
    digitize_idx  = np.digitize(X, bins)
    running_med = [np.median(Y[digitize_idx==k]) for k in range(total_bins)]
    running_std = [np.std(Y[digitize_idx==k]) for k in range(total_bins)]
    return bins-delta/2, running_med, running_std


def overlap(a, b):
    """Computes the overlap between the ranges of a and b"""
    err_msg = 'The inputs "a" and "b" have to be a list/tuple of length 2.'
    if not isinstance(a, (tuple, set, list)) or not isinstance(a, (tuple, set, list)):
        raise TypeError(err_msg)
    if len(a)!=2 or len(b)!=2:
        raise ValueError(err_msg)
    # Sort list/tuples
    a = sorted(a)
    b = sorted(b)
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def mode_continuous(contious_data, n_bins=100):
    # 1) bin data
    hist, bin_edges = np.histogram(contious_data, bins=n_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    # 2) calculte mode
    mode = bin_centers[np.argmax(hist)]
    return mode


grid_points_1d = lambda lim, delta: np.arange(min(lim), max(lim) + delta, delta)

def hexagonal_grid_points(scaling_const, xlim, ylim):
    """
    :param scaling_const: difference between points in x axes
    :param xlim, ylim: Grid-limits in x- and y-direction (tuple)
    :return: Stacked 2D vectors of hexagonal lattice
    """
    scaling_const = float(scaling_const)  # cast as float, otherwise xv & yv might be integer arrays
    ratio = np.sqrt(3) / 2
    x_step = scaling_const
    y_step = scaling_const * ratio
    # Get grid sides
    x_points = grid_points_1d(xlim, x_step)
    y_points = grid_points_1d(ylim, y_step)

    xv, yv = np.meshgrid(x_points, y_points, sparse=False, indexing='xy')
    # Indent every two lines by half the width
    xv[::2, :] += ratio / 2
    positions = np.vstack([xv.ravel(), yv.ravel()])
    return positions.T


def hist2data(entries_per_bin: np.ndarray):
    """Input a histogram and return data"""
    shifted_neighbors = (entries_per_bin-np.min(entries_per_bin)+2).astype(int) # minimum of 2 entries per bin
    x = np.arange(shifted_neighbors.size)+1
    hist_nn = np.repeat(x, shifted_neighbors).astype(np.float32)  # create histogram data
    # Add random noise to each entry
    hist_nn += np.random.uniform(size=hist_nn.size)
    return hist_nn


def distance2projectedpoint(a, b, p):
    """Project point p onto line spanned by a & b
    Returns distance a-p on line a-b
    """
    ap = p-a
    ab = b-a
    return np.dot(ap, ab)/np.sqrt(np.dot(ab, ab))


def plot_gmm_results(gmm, rho, cols=['tab:blue', 'tab:orange']):
    xmin, xmax = np.min(rho), np.max(rho)
    nb, bins, _ = plt.hist(rho, bins=100, histtype='step', color='k')
    integral = np.trapz(nb, (bins[:-1]+bins[1:])/2)
    for mu, w, var, col in zip(gmm.means_, gmm.weights_, gmm.covariances_, cols):
        sigma = np.sqrt(var[0])
        xnormal = np.linspace(xmin, xmax, 100)
        plt.plot(xnormal, stats.norm.pdf(xnormal, mu, sigma)*w*integral, color=col)
    return bins


def loguniform(value, lower_bound=1e-2, upper_bound=1e0):
    """
    value: sample between 0 and 1
    loguniform.ppf: inverse of CDF (https://en.wikipedia.org/wiki/Inverse_transform_sampling)
    """
    return stats.loguniform.ppf(value, lower_bound, upper_bound, loc=0, scale=1)


def transform2range(value, range_low, range_up):
    """Transform 0-1 range (from uniform sample) to new range"""
    return value * (range_up - range_low) + range_low


def mad_based_outlier(points, thresh=3):
    """1d outlier detection based on symmetric MAD"""
    if isinstance(points, (list, tuple)):
        points = np.array(points)
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



