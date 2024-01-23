import numpy as np


class XDSingleCluster:
    """Class to compute the bulk velocity of a single cluster using XD.
    Usage example:
    >>> from SigMA.NoiseRemoval.xd_special import XDSingleCluster
    >>> xds = XDSingleCluster().fit(X, Xerr)
    """
    def __init__(self, max_iter=100, tol=1e-2):
        self.max_iter = max_iter
        self.tol = tol
        self.mu = None
        self.V = None

    def fit(self, X, Xerr):
        mu = np.mean(X, axis=0)
        V = np.cov(X, rowvar=False)[np.newaxis, :]

        logL = self.multiple_logpdfs(X, mu, Xerr + V).sum()
        for i in range(self.max_iter):
            mu, V = self.EM_step(X, Xerr, mu, V)
            # Check if logL has converged
            logL_next = self.multiple_logpdfs(X, mu, Xerr + V).sum()
            if logL_next < logL + self.tol:
                break
            logL = logL_next

        self.mu = mu
        self.V = V
        return self

    @staticmethod
    def EM_step(X, Xerr, mu, V):
        n_samples, n_features = X.shape
        Xfit = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]
        w_m = Xfit - mu
        T = Xerr + V
        # ------------------------------------------------------------
        #  compute inverse of each covariance matrix T
        Tinv = np.array([np.linalg.inv(T[i]) for i in range(T.shape[0])]).reshape(T.shape)
        # # ------------------------------------------------------------
        # #  E-step:
        q = np.ones((n_samples, 1))
        tmp = np.sum(Tinv * w_m[:, :, np.newaxis, :], -1)
        b = mu + np.sum(V * tmp[:, :, np.newaxis, :], -1)
        tmp = np.sum(Tinv[:, :, :, :, np.newaxis] * V[:, np.newaxis, :, :], -2)
        B = V - np.sum(V[:, :, :, np.newaxis] * tmp[:, :, np.newaxis, :, :], -2)
        # ------------------------------------------------------------
        #  M-step:
        #   compute alpha, m, V
        qj = np.array([n_samples])
        mu_new = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis]
        m_b = mu_new - b
        tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
        tmp += B
        tmp *= q[:, :, np.newaxis, np.newaxis]
        V_new = tmp.sum(0) / qj[:, np.newaxis, np.newaxis]

        return mu_new, V_new

    @staticmethod
    def multiple_logpdfs(x, means, covs):
        """Fast computation of multivariate normal log PDF over multiple sets of covariances.
        Courtesy of: https://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
        means can be a 1D or 2D array, covs must be a 3D array.
        """
        # NumPy broadcasts `eigh`.
        vals, vecs = np.linalg.eigh(covs)
        # Compute the log determinants across the second axis.
        logdets = np.sum(np.log(vals), axis=1)
        # Invert the eigenvalues.
        valsinvs = 1. / vals
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
