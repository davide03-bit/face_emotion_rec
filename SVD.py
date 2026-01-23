from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import numpy as np

def SVD(X):
    U, S, V= svd(X)
    return U, S, V

def explained_variance_ratio(X, exp_var_optimal):
    _, S, Vt= SVD(X)
    eigenvalues = S ** 2

    total_variance = np.sum(eigenvalues)

    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    k = np.argmax(cumulative_variance_ratio >= exp_var_optimal) + 1

    Vt_truncated = Vt[:k, :]

    return Vt_truncated.T



