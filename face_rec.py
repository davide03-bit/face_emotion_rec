import pandas as pd
from sklearn.metrics import classification_report
from preprocessing import data_split
from PCA import Classification, Classification2
from plot import plot_eigenfaces, plot_confusion_matrix_heatmap
import matplotlib.pyplot as plt
import numpy as np
import time

np.random.seed(42)

path = "ckextended.csv"

df = pd.read_csv(path)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(df.head())
print(df.shape)
df.info()
print(df.describe())

print(f"Emotion distribution:\n{df['emotion'].value_counts().sort_index()}")

# list of available emotions in the dataset
emotions = { 
    0:"Anger",
    1:"Disgust",
    2:"Fear",
    3:"Happiness",
    4:"Sadness",
    5:"Surprise",
    6:"Neutral",
    7:"Contempt"
}

df['emotion_name'] = df['emotion'].copy()

for num, val in emotions.items():
    df['emotion_name'].replace(num, val, inplace=True)
print(df.head())

X_train, y_train, X_test, y_test = data_split(df)

exp_var_optimal = 0.95

print("CLASSIFICATORE SVM CON PCA")

start = time.perf_counter()
# V_truncated= explained_variance_ratio(X_train, exp_var_optimal)
classification = Classification(X_train, y_train, exp_var_optimal)
end = time.perf_counter()
print(f'\nTEMPO DI ADDESTRAMENTO PCA+SVM: {end-start}')
y_pred = classification.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
plot_confusion_matrix_heatmap(y_test, y_pred, emotions)

print("CLASSIFICATORE SVM SENZA PCA")

start = time.perf_counter()
classification2 = Classification2(X_train, y_train)
end = time.perf_counter()
print(f'TEMPO DI ADDESTRAMENTO SVM: {end-start}')
y_pred2= classification2.predict(X_test)
print(classification_report(y_test, y_pred2, zero_division=0))
plot_confusion_matrix_heatmap(y_test, y_pred2, emotions)

# Ricostruzione immagine non rumorosa
# Prendi un'immagine non rumorosa
#img= X_train[0]

# Proietta nello spazio latente

#latent_vector = np.dot(img, V_truncated)

#reconstructed_img = np.dot(latent_vector, V_truncated.T)

#plt.figure(figsize=(10, 4))
#plt.subplot(1, 3, 1)
#plt.imshow(X_train[0].reshape(48, 48), cmap='gray')
#plt.title("Originale")

#plt.subplot(1, 3, 3)
#plt.imshow(reconstructed_img.reshape(48, 48), cmap='gray')
#plt.title("Immagine ricostruita")
#plt.show()


# aggiunta rumore pca+svm

var_levels = [10, 30, 60]

for var in var_levels:
    print(f"\n--- ANALISI COMPARATIVA: VARIANZA {var} ---")

    # Reset del seme per ogni livello di varianza 
    rng = np.random.default_rng(42)
    rumore = rng.normal(0, var, X_test.shape)
    X_test_da_usare = X_test + rumore

    # PCA + SVM
    print("Risultati PCA + SVM:")
    y_pred_pca = classification.predict(X_test_da_usare)
    print(classification_report(y_test, y_pred_pca, zero_division=0))

    # SVM SENZA PCA
    print("Risultati SVM Standard:")
    y_pred_svm = classification2.predict(X_test_da_usare)
    print(classification_report(y_test, y_pred_svm, zero_division=0))



#n_esempi = 5
#indices = rng.choice(X_test.shape[0], n_esempi, replace=False)

#plt.figure(figsize=(15, 5))

#for i, idx in enumerate(indices):
    # Immagine Originale
    #plt.subplot(2, n_esempi, i + 1)
    #original_img = X_test[idx].reshape(48, 48)  # Reshape per tornare immagine
    #plt.imshow(original_img, cmap='gray')
    #plt.title("Originale")
    #plt.axis('off')

    # Immagine con Rumore
    #plt.subplot(2, n_esempi, i + 1 + n_esempi)
    #noisy_img = X_test_da_usare[idx].reshape(48, 48)
    #plt.imshow(noisy_img, cmap='gray')
    #plt.title(f"Rumore {VARIANZA_RUMORE}")
    #plt.axis('off')

#plt.tight_layout()
#plt.show()




#RICOSTRUZIONE IMMAGINE RUMOROSA
# Prendi un'immagine rumorosa
#img_noisy = X_test_da_usare[0]

# Proietta nello spazio latente
#latent_vector = np.dot(img_noisy, V_truncated)
# X_rec = Z * V_T
#reconstructed_img = np.dot(latent_vector, V_truncated.T)

#plt.figure(figsize=(10, 4))
#plt.subplot(1, 3, 1)
#plt.imshow(X_test[0].reshape(48, 48), cmap='gray')
#plt.title("Originale (Pulito)")

#plt.subplot(1, 3, 2)
#plt.imshow(img_noisy.reshape(48, 48), cmap='gray')
#plt.title(f"Input Rumoroso (Var={VARIANZA_RUMORE})")

#plt.subplot(1, 3, 3)
#plt.imshow(reconstructed_img.reshape(48, 48), cmap='gray')
#plt.title("Ciò che vede la SVM (Dopo PCA)")
#plt.show()

