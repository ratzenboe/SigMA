import numpy as np


def multiple_logpdfs(x, means, covs):
    """Compute multivariate normal log PDF over multiple sets of parameters.
    means can be a single mean or an array of means.
    Courtesy of: https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
    """
    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)
    # Compute the log determinants across the second axis.
    logdets = np.sum(np.log(vals), axis=1)
    # Invert the eigenvalues.
    valsinvs = 1./vals
    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us = vecs * np.sqrt(valsinvs)[:, None]
    devs = x - means
    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs = np.einsum('ni,nij->nj', devs, Us)
    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas = np.sum(np.square(devUs), axis=1)
    # Compute and broadcast scalar normalizers.
    dim = len(vals[0])
    log2pi = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + mahas + logdets)
