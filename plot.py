import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_eigenfaces(V, image_shape=(48, 48), n_plots=10):
    plt.figure(figsize=(15, 6))
    for i in range(min(n_plots, V.shape[1])):
        plt.subplot(2, 5, i + 1)
        # V[:, i] è l'i-esima componente principale
        # Reshape per visualizzarla come immagine
        face = V[:, i].reshape(image_shape)
        plt.imshow(face, cmap='gray')
        plt.title(f'Componente {i + 1}')
        plt.axis('off')
    plt.suptitle('Top Eigenfaces (Le feature imparate)', fontsize=16)
    plt.show()


def plot_confusion_matrix_heatmap(y_test, y_pred, labels_map):
    cm = confusion_matrix(y_test, y_pred)
    # Converti le label numeriche in nomi
    names = [labels_map[i] for i in sorted(labels_map.keys()) if i in np.unique(y_test)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title('Matrice di Confusione')
    plt.ylabel('Vera Emozione')
    plt.xlabel('Emozione Predetta')
    plt.show()