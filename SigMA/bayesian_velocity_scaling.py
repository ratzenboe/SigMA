from sklearn.neighbors import NearestNeighbors
from arviz.stats import hdi
from scipy.interpolate import griddata, interp1d
from scipy.signal import savgol_filter
from scipy.stats import norm
import numpy as np


def produce_kde(x, posterior_predictive, x_range=[0, 300]):
    # Get 1 sigma confidence intervals
    hdi_data = hdi(posterior_predictive, hdi_prob=0.68, circular=False, multimodal=False)
    # smooth data
    smooth_kwargs = {}
    smooth_kwargs.setdefault("window_length", 100)
    smooth_kwargs.setdefault("polyorder", 2)
    x_data = np.linspace(x.min(), x.max(), 200)
    x_data[0] = (x_data[0] + x_data[1]) / 2
    hdi_interp = griddata(x, hdi_data, x_data)
    y_data = savgol_filter(hdi_interp, axis=0, **smooth_kwargs)
    # Compute std deviations
    y_means = np.mean(y_data, axis=1)
    y_std_devs = np.diff(y_data, axis=1) / 2
    # Get x-limit
    isin_range = (min(x_range) <= x_data) & (x_data <= max(x_range))
    # Compute sum of probabilies (kde)
    y_kde_range = np.linspace(0, 40, 300)
    y_kde = np.zeros_like(y_kde_range)
    for y_mu, y_std in zip(y_means[isin_range], y_std_devs[isin_range]):
        # compute
        y_kde += norm(y_mu, y_std).pdf(y_kde_range)
    return y_kde_range, y_kde


def ecdf(x):
    # normalize X to sum to 1
    x = x / np.sum(x)
    return np.cumsum(x)


def scale_factors(x, posterior_predictive, x_range=[0, 300], n_samples=10):
    """Compute scaling factors for a given distance range
    Funciton needs information on posterior predictive and corresponding x locations -- needs to be loaded from file
    """
    x_kde_input, y_kde = produce_kde(x, posterior_predictive, x_range=x_range)

    f_ecdf = interp1d(x_kde_input, ecdf(y_kde))
    x_grid = np.linspace(0, 40, 1000)
    # Use NN in y to get x values
    nn = NearestNeighbors(n_neighbors=1).fit(f_ecdf(x_grid).reshape(-1, 1))
    # Get equidistant samples in y
    samples_quantile = np.linspace(0.05, 0.95, n_samples).reshape(-1, 1)
    # Get indices of closest points in y
    _, indices = nn.kneighbors(samples_quantile)
    # return corresponding x values
    return x_grid[indices.ravel()]
