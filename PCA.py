from sklearn.decomposition import PCA
import numpy as np

def PCA_sklearn(X, V):
    pca= PCA(n_components= len(V))
    X= pca.fit_transform(X)
    return X


def PCA(X, V):
    Z= np.dot(X, V)
    return Z
