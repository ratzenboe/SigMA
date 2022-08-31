import numpy as np
import scipy.stats as stats
from sklearn.mixture import GaussianMixture


def gmm_cut(data2fit, n_components=2):
    """Fit a gaussian mixture to the given 1D data and output classification parameters"""
    if data2fit.shape[0] <= 10:
        return None, None, None, None, None, False

    max_value_per_class = []
    n_trials = 0
    while len(max_value_per_class) != n_components:  # Sometimes a GMM will only output 1 component -> check for that
        # Train the GMM on the input data
        gmm = GaussianMixture(n_components=n_components).fit(data2fit.reshape(-1, 1))
        gmm_lbls = gmm.predict(data2fit.reshape(-1, 1))   # implement also probability estimate predict_proba(X)
        # Get maximum value per class
        max_value_per_class = [np.max(data2fit[gmm_lbls == li]) for li in np.unique(gmm_lbls)]
        # Fail save
        if n_trials > 50:
            print(f'Input unable to be clustered into {n_components} classes')
            return None, None, None, None, None, False
        n_trials += 1

    # -> Sort in descending order by maximum density, then take the second highest one
    low_density_cluster = np.argsort(max_value_per_class)[::-1][1]
    # Maximium value of 2nd highest value/denisty cluster
    th = np.max(data2fit[gmm_lbls == low_density_cluster])
    # Maximum value/density class is the one with larger values than the maximum of the 2nd highest class values
    cluster_labels = data2fit > th

    # Simple contamination estimate: Contamination, or false positive rate is estimated via the probability that
    # the background density is higher than the threshold "th"
    arg_bg_model = np.argmin(gmm.means_.flatten())
    # Get details of Gaussians
    mu_bg = gmm.means_.flatten()[arg_bg_model]
    var_bg = gmm.covariances_.flatten()[arg_bg_model]
    sigma_bg = np.sqrt(var_bg)
    # Compute contamination fraction
    contamination_fraction = 1 - stats.norm.cdf(th, mu_bg, sigma_bg)
    # Simple completeness estimate via the false negatives
    arg_sig_model = np.argmax(gmm.means_.flatten())
    # Get details of Gaussians
    mu_sig = gmm.means_.flatten()[arg_sig_model]
    var_sig = gmm.covariances_.flatten()[arg_sig_model]
    sigma_sig = np.sqrt(var_sig)
    completeness_fraction = 1 - stats.norm.cdf(th, mu_sig, sigma_sig)

    return gmm, cluster_labels, th, contamination_fraction, completeness_fraction, True

