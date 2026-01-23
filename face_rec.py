import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from preprocessing import data_split

path = "/home/davide/Scaricati/ckextended.csv"

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

X_train, y_train, X_validation, y_validation, X_test, y_test = data_split(df)

print(X_train)