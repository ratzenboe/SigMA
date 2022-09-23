import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture


def gmm_cut(X, n_components=2):
    """Fit a gaussian mixture to the given 1D data and output classification parameters"""
    if X.shape[0] <= 10:
        return None, None, None, None, None, False
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # If we need to compte the new density
    compute_k = not isinstance(n_components, int)
    if not compute_k:
        if (n_components > 5) or (n_components < 2):
            compute_k = True
    if compute_k:
        n_components, gmm = find_optimal_number_of_components(X)
    else:
        gmm = GaussianMixture(n_components=n_components, n_init=2).fit(X)
    # Get labels
    gmm_lbls = gmm.predict(X)   # implement also probability estimate predict_proba(X)

    th = find_threshold(
        X, gmm_lbls,
        mus=gmm.means_.flatten(), var=gmm.covariances_.flatten(), w=gmm.weights_.flatten()
    )
    # Maximum value/density class is the one with larger values than the maximum of the 2nd highest class values
    cluster_labels = X.flatten() > th
    # Compute contamination and completeness
    contamination_fraction = estimate_contamination(gmm, th)
    completeness_fraction = estimate_completeness(gmm, th)

    return gmm, cluster_labels, th, contamination_fraction, completeness_fraction, True


def find_threshold(data, labels, mus, var, w):
    sorted_by_var = np.argsort(var)
    sorted_by_weight = np.argsort(w)[::-1]
    sorted_by_mean = np.argsort(mus)[::-1]
    # Typically, largest component or smallest variance is background
    # If largest component has also the highest mean value, then we take second largest
    bg_idx = sorted_by_var[0]
    # If the smallest variance is not the lowest mean, we make a few tests
    if sorted_by_var[0] != sorted_by_mean[-1]:
        if sorted_by_weight[0] == sorted_by_var[0]:
            bg_idx = sorted_by_weight[0]
        elif sorted_by_weight[0] == sorted_by_mean[-1]:
            bg_idx = sorted_by_weight[0]
        elif sorted_by_var[1] == sorted_by_mean[-1]:
            bg_idx = sorted_by_var[1]
        else:
            bg_idx = sorted_by_mean[1]
    # Maximium value of 2nd highest value/denisty cluster
    th = np.max(data[labels == bg_idx])
    return th


def estimate_contamination(gmm, th):
    w = gmm.weights_.flatten()
    mus = gmm.means_.flatten()
    sigmas = np.sqrt(gmm.covariances_.flatten())
    # Background components have means less than threshold
    bg = mus < th
    # Fraction of background that extends into signal region
    bg_in_signal, all_in_signal = 0, 0
    for is_bg, p_i, mu_i, sigma_i in zip(bg, w, mus, sigmas):
        pp_i = p_i * (1 - stats.norm.cdf(th, mu_i, sigma_i))
        all_in_signal += pp_i
        if is_bg:
            bg_in_signal += pp_i

    if all_in_signal == 0:
        contamination = 1
    else:
        contamination = bg_in_signal/all_in_signal
    return contamination


def estimate_completeness(gmm, th):
    w = gmm.weights_.flatten()
    mus = gmm.means_.flatten()
    sigmas = np.sqrt(gmm.covariances_.flatten())
    # Background components have means less than threshold
    signal = mus > th
    # completeness_fraction = 1 - stats.norm.cdf(th, mu_sig, sigma_sig)
    isin_signal = 0
    for is_sig, p_i, mu_i, sigma_i in zip(signal, w, mus, sigmas):
        if is_sig:
            pp_i = p_i * (1 - stats.norm.cdf(th, mu_i, sigma_i))
            isin_signal += pp_i
    # In case no signal is found
    if np.sum(signal) == 0:
        completeness = 0
    else:
        # Needs to be normalized with the sum of probabilities
        completeness = isin_signal/np.sum(w[signal])
    return completeness


def find_optimal_number_of_components(X, min_k=2, max_k=5):
    minimizer = []
    gmm_list = []
    nb_components = np.arange(min_k, max_k+1)
    for k in nb_components:
        gmm = GaussianMixture(n_components=k, n_init=2).fit(X)
        gmm_list.append(gmm)
        bic = gmm.bic(X)
        minimizer.append(bic)
    min_bic = np.argmin(minimizer)
    return nb_components[min_bic], gmm_list[min_bic]
