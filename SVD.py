from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

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

   # plot_explained_variance(S, total_variance)
   # plot_reconstruction(X[10], Vt.T, k)
    #plot_scree(S)

    return Vt_truncated.T

""""
def plot_explained_variance(S, total_variance):
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


def plot_reconstruction(X_image, V, k, image_shape=(48, 48)):
    V_k = V[:, :k]  # Forma: (pixels, k)
    weights = np.dot(X_image, V_k)
    X_reconstructed = np.dot(weights, V_k.T)

    plt.figure(figsize=(8, 4))

    # Originale
    plt.subplot(1, 2, 1)
    plt.imshow(X_image.reshape(image_shape), cmap='gray')
    plt.title('Originale (Standardizzato)')
    plt.axis('off')

    # Ricostruita
    plt.subplot(1, 2, 2)
    plt.imshow(X_reconstructed.reshape(image_shape), cmap='gray')
    plt.title(f'Ricostruzione con k={k}')
    plt.axis('off')

    plt.show()


def plot_scree(S):
    eigenvalues = S ** 2
    k = len(eigenvalues)
    x = np.arange(1, k + 1)
    y = eigenvalues[:k]

    #Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(x, y, 'o-', linewidth=2, color='blue', markersize=6)

    plt.title(f'Primi {k} Autovalori')
    plt.xlabel('Componente Principale (PC)')
    plt.ylabel('Autovalore')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.show()
"""
