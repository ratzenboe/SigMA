from scipy.special import logsumexp
import numpy as np
from sklearn.covariance import MinCovDet


class XDOutlier:
    """Class to compute the bulk velocity of a single cluster using XD.
    Usage example:
    >>> from SigMA.NoiseRemoval.xd_outlier import XDOutlier
    >>> xds = XDOutlier().fit(X, Xerr)
    >>> mu, V = xds.min_entropy_component()
    """

    def __init__(self, max_iter=200, tol=1e-3, max_alpha=0.95):
        self.max_iter = max_iter
        self.tol = tol
        self.max_alpha = max_alpha
        self.mu = None
        self.V = None
        self.alpha = None

    def set_max_alpha(self, max_alpha):
        """Set the maximum alpha value for the outlier component."""
        self.max_alpha = max_alpha

    def estimate_init_params(self, X, rv_not_nan=None):
        """Estimate the bulk velocity of the cluster."""
        _, n_features = X.shape
        # Compute inital estiamate for mean velocity and covariance matrix using MCD
        if rv_not_nan is not None:
            mcd = MinCovDet().fit(X[rv_not_nan])
            # Extract mean and covariance matrix
            mu = mcd.location_
            V = mcd.covariance_
        else:
            # Extract mean and covariance matrix
            mu = np.mean(X, axis=0)
            V = np.cov(X, rowvar=False)

        # Add second component for outlier
        mu = np.vstack((mu, np.zeros(n_features)))
        V = np.r_[V[np.newaxis, :, :], np.eye(n_features)[np.newaxis, :, :] * 1e10]
        alpha = np.array([0.8, 0.2])

        return mu, V, alpha

    def fit(self, X, Xerr, rv_not_nan=None):
        mu, V, alpha = self.estimate_init_params(X, rv_not_nan=rv_not_nan)
        # Start EM algorithm
        logL = self.logpdf_gmm(X, Xerr, mu, V, alpha)
        for i in range(self.max_iter):
            mu, V, alpha = self.EM_step_outlier(X, Xerr, mu, V, alpha)
            # Check if logL has converged
            logL_next = self.logpdf_gmm(X, Xerr, mu, V, alpha)
            if logL_next < logL + self.tol:
                break
            logL = logL_next
        self.mu = mu
        self.V = V
        self.alpha = alpha
        return self

    def EM_step_outlier(self, X, Xerr, mu_fit, V_fit, alpha):
        n_samples, n_features = X.shape
        if np.max(alpha) > self.max_alpha:
            # Cluster component might not fit properly if outlier component can vanish
            idx_min = np.argmin(alpha)
            idx_max = np.argmax(alpha)
            alpha[idx_min] = 1- self.max_alpha
            alpha[idx_max] = self.max_alpha
        # Add outlier component: mu = 0, V = 1e10
        X_fit = X[:, np.newaxis, :]
        w_m = X_fit - mu_fit
        T = Xerr[:, np.newaxis, :, :] + V_fit
        # ------------------------------------------------------------
        #  compute inverse of each covariance matrix T
        Tshape = T.shape
        T = T.reshape([n_samples * 2,
                       n_features, n_features])
        Tinv = np.array([np.linalg.inv(T[i])
                         for i in range(T.shape[0])]).reshape(Tshape)
        # ------------------------------------------------------------
        #  evaluate each mixture at each point
        # N = np.exp(log_multivariate_gaussian(X_fit, mu_fit, T, Vinv=Tinv))
        N1 = self.multiple_logpdfs(X, mu_fit[0], Xerr + V_fit[0])
        N2 = self.multiple_logpdfs(X, mu_fit[1], Xerr + V_fit[1])
        N = np.exp(np.c_[N1, N2])
        # # ------------------------------------------------------------
        # #  E-step:
        #   compute q_ij, b_ij, and B_ij
        q = (N * alpha) / np.dot(N, alpha)[:, None]
        tmp = np.sum(Tinv * w_m[:, :, np.newaxis, :], -1)
        b = mu_fit + np.sum(V_fit * tmp[:, :, np.newaxis, :], -1)
        tmp = np.sum(Tinv[:, :, :, :, np.newaxis] * V_fit[:, np.newaxis, :, :], -2)
        B = V_fit - np.sum(V_fit[:, :, :, np.newaxis] * tmp[:, :, np.newaxis, :, :], -2)
        # ------------------------------------------------------------
        #  M-step:
        #   compute alpha, m, V
        qj = q.sum(0)
        alpha_new = qj / n_samples

        mu_new = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis]
        m_b = mu_new - b
        tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
        tmp += B
        tmp *= q[:, :, np.newaxis, np.newaxis]
        V_new = tmp.sum(0) / qj[:, np.newaxis, np.newaxis]

        return mu_new, V_new, alpha_new

    def min_entropy_component(self):
        """Return the component with the lowest entropy."""
        V1, V2 = self.V
        mu1, mu2 = self.mu
        # Compute entropy
        S1 = np.linalg.slogdet(V1)[1]
        S2 = np.linalg.slogdet(V2)[1]
        # Return component with lowest entropy
        if S1 < S2:
            return mu1, V1
        else:
            return mu2, V2

    def logpdf_gmm(self, x, xerrs, means, covs, alpha):
        logL = 0
        for i in range(len(alpha)):
            logL += logsumexp(np.log(alpha[i]) + self.multiple_logpdfs(x, means[i], xerrs + covs[i]))
        return logL

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
