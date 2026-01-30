import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from preprocessing import data_split
from SVD import SVD, explained_variance_ratio
from PCA import PCA, Classification
from plot import plot_eigenfaces, plot_confusion_matrix_heatmap

path = "ckextended.csv"

df = pd.read_csv(path)
df = df.sample(frac=1).reset_index(drop=True)

print(df.head())
print(df.shape)
df.info()
print(df.describe())

print(f"Emotion distribution:\n{df['emotion'].value_counts().sort_index()}")


#list of available emotions in the dataset
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

exp_var_optimal= 0.95
V_truncated= explained_variance_ratio(X_train, exp_var_optimal)

y_pred= Classification(X_train, X_test, y_train, V_truncated)
print(classification_report(y_test, y_pred))

plot_eigenfaces(V_truncated, image_shape=(48, 48))
plot_confusion_matrix_heatmap(y_test, y_pred, emotions)
