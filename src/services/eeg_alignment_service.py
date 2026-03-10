import numpy as np
from scipy.linalg import sqrtm, inv


class EEGAlignmentService:
    @staticmethod
    def euclidean_alignment(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply Euclidean Alignment to reduce inter-subject variability.

        He & Wu, IEEE Trans. Biomedical Engineering, 2020.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_times)

        Returns
        -------
        X_aligned : same shape, whitened trials
        R_inv_sqrt : (n_channels, n_channels) matrix to reuse at test time
        """
        n_trials, n_ch, n_t = X.shape

        R_mean = np.zeros((n_ch, n_ch))
        for i in range(n_trials):
            R_mean += X[i] @ X[i].T / n_t
        R_mean /= n_trials

        # Regularize for rank-deficient data (e.g. after CAR re-referencing)
        R_mean += np.eye(n_ch) * 1e-7

        R_inv_sqrt = inv(sqrtm(R_mean)).real.astype(np.float64)

        X_aligned = np.zeros_like(X)
        for i in range(n_trials):
            X_aligned[i] = R_inv_sqrt @ X[i]

        return X_aligned, R_inv_sqrt
