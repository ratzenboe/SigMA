from scipy.special import logsumexp
import numpy as np



from scipy.special import logsumexp
import numpy as np



class XDOutlier:
    """Class to compute the bulk velocity of a single cluster using XD.
    Usage example:
    >>> from SigMA.NoiseRemoval.xd_special import XDOutlier
    >>> xds = XDOutlier().fit(X, Xerr)
    >>> mu, V = xds.min_entropy_component()
    """

    def __init__(self, max_iter=200, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.mu = None
        self.V = None
        self.alpha = None

    def fit(self, X, Xerr):
        _, n_features = X.shape
        mu = np.mean(X, axis=0)
        V = np.cov(X, rowvar=False)
        # Add second component for outlier
        mu = np.vstack((mu, np.zeros(n_features)))
        V = np.r_[V[np.newaxis, :, :], np.eye(n_features)[np.newaxis, :, :] * 1e10]
        alpha = np.array([0.8, 0.2])

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
        if np.max(alpha) > 0.9:
            # Cluster component might not fit properly if outlier component can vanish
            alpha = np.array([0.9, 0.1])
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