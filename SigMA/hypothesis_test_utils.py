import numpy as np
from scipy.stats import chi2, cauchy
from scipy.interpolate import UnivariateSpline


def global_pvalue_hmp(pvalue_list):
    """Implements the The harmonic mean p-value (HMP). It is a method for performing a combined test
    of the null hypothesis that no p-value is significant (Wilson, 2019).
    It is more robust to dependence between p-values than Fisher’s (1934) method, making it more broadly applicable
    :param pvalue_list: List of p-values from individual tests
    :param weights: weight list with length of pvalue_list and must sum to 1
    """
    # Some checks on the passed weights
    # if isinstance(weights, (list, np.ndarray)):
    #     if (len(weights) != len(pvalue_list)) or (np.abs(1-np.sum(weights) > 1e-5)):
    #         weights = np.ones(len(pvalue_list), dtype=np.float) / len(pvalue_list)
    # else:
    #     weights = np.ones(len(pvalue_list), dtype=np.float) / len(pvalue_list)

    weights = np.ones(len(pvalue_list)) / len(pvalue_list)
    # Convert to numpy (just to be sure)
    pvalue_list = np.asarray(pvalue_list)
    # Compute HMP
    pvalue_hmp = np.sum(weights) / np.sum(weights/pvalue_list)
    return pvalue_hmp


def global_pvalue_fisher(pvalue_list):
    """Fisher’s combination test. If the p_i are uniformly distributed in [0,1],
    then the negative logarithm follows an exponential distribution: -log pi ~ Exp(1).
     The test statistic  t  then follows a 2 distribution with 2n degrees of freedom:
        t = -2 in log pi  ~ chi2_n2
    The global hypothesis test is performed by evaluating if  t  is exceedingly large.
    """
    pvalue_list = np.array(pvalue_list)
    result = np.where(pvalue_list > 0.0000000001, pvalue_list, -23.0259)
    result = np.log(result, out=result, where=result > 0)

    t = -2*np.sum(result)
    degrees_of_freedom = 2 * pvalue_list.size
    return chi2.sf(t, degrees_of_freedom)

def cauchy_combination_test(pvalue_list):
    """The Cauchy combination test is a robust alternative to Fisher’s combination test.
    The test statistic under the Null is standard Cauchy distributed (i.e. with location 0 and scale 1).
    See: doi.org/10.1080/01621459.2018.1554485
    """
    pvalue_list = np.array(pvalue_list)
    t = np.mean(np.tan((0.5 - pvalue_list)*np.pi))
    # Compute p-value
    return cauchy.sf(t, 0, 1)

def global_pvalue_bonferroni(pvalue_list):
    """Implements the Bonferroni correction"""
    return min(pvalue_list)*len(pvalue_list)


def _ecdf(x):
    """no frills empirical cdf used in correct_pvals_benjamini_hochberg"""
    nobs = len(x)
    return np.arange(1, nobs + 1) / float(nobs)


def correct_pvals_benjamini_hochberg(pvals, alpha=0.05):
    pvals_sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvals, pvals_sortind)
    ecdffactor = _ecdf(pvals_sorted)

    reject = pvals_sorted <= ecdffactor * alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True
        pval_last_reject = pvals_sorted[reject][-1]
        # pval_first_nonreject = pvals_sorted[~reject][0]
        # return (pval_last_reject + pval_first_nonreject)/2
        return pval_last_reject + 1e-5
    else:
        return pvals_sorted[0] + 1e-5


def qvalue(pvals, threshold=0.05):
    """Function from https://github.com/puolival/multipy/blob/master/multipy/fdr.py
    Function for estimating q-values from p-values using the Storey-
    Tibshirani q-value method (2003).
    Input arguments:
    ================
    pvals       - P-values corresponding to a family of hypotheses.
    threshold   - Threshold for deciding which q-values are significant.
    Output arguments:
    =================
    significant - An array of flags indicating which p-values are significant.
    qvals       - Q-values corresponding to the p-values.
    """

    """Count the p-values. Find indices for sorting the p-values into
    ascending order and for reversing the order back to original."""
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    rev_ind = np.argsort(ind)
    pvals = pvals[ind]

    # Estimate proportion of features that are truly null.
    kappa = np.arange(0, 0.96, 0.01)
    pik = [np.sum(pvals > k) / (m*(1-k)) for k in kappa]
    cs = UnivariateSpline(kappa, pik, k=3, s=None, ext=0)
    pi0 = float(cs(1.))

    # The smoothing step can sometimes converge outside the interval [0, 1].
    # This was noted in the published literature at least by Reiss and
    # colleagues [4]. There are at least two approaches one could use to
    # attempt to fix the issue:
    # (1) Set the estimate to 1 if it is outside the interval, which is the
    #     assumption in the classic FDR method.
    # (2) Assume that if pi0 > 1, it was overestimated, and if pi0 < 0, it
    #     was underestimated. Set to 0 or 1 depending on which case occurs.
    # Here we have chosen the first option, since it is the more conservative
    # one of the two.

    if (pi0 < 0) or (pi0 > 1):
        pi0 = 1
        print('Smoothing estimator did not converge in [0, 1]')

    # Compute the q-values.
    qvals = np.zeros(np.shape(pvals))
    qvals[-1] = pi0*pvals[-1]
    for i in np.arange(m-2, -1, -1):
        qvals[i] = min(pi0*m*pvals[i]/float(i+1), qvals[i+1])

    # Test which p-values are significant.
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind] = qvals<threshold

    # Order the q-values according to the original order of the p-values
    qvals = qvals[rev_ind]
    return significant, qvals
