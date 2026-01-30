from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np

def SVD(X):
    _, S, V= svd(X)
    return S, V

def explained_variance_ratio(X, exp_var_optimal):
    S, Vt= SVD(X)
    eigenvalues = S ** 2

    total_variance = np.sum(eigenvalues)

    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    k = np.argmax(cumulative_variance_ratio >= exp_var_optimal) + 1

    Vt_truncated = Vt[:k, :]

    plot_explained_variance(S, total_variance)

    return Vt_truncated.T


def plot_explained_variance(S, total_variance):
    """
    Mostra quanta varianza spiegano le componenti (Scree Plot).
    S: array dei valori singolari (da SVD.py)
    """
    eigenvalues = S ** 2
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, linewidth=2)
    plt.xlim(0, 200)
    plt.title('Varianza Spiegata Cumulativa')
    plt.xlabel('Numero di Componenti')
    plt.ylabel('Varianza Spiegata (%)')
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
    plt.legend()
    plt.show()

