import numpy as np
from sklearn.preprocessing import StandardScaler

def pixels_to_matrix(dataframe):
    pixels = dataframe['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    X = np.vstack(pixels)
    return X

def data_split(dataframe):
    mask_train = (dataframe['Usage'] == 'Training') | (dataframe['Usage'] == 'PrivateTest')
    mask_test = dataframe['Usage'] == 'PublicTest'

    X_train_raw = pixels_to_matrix(dataframe[mask_train])
    X_test_raw = pixels_to_matrix(dataframe[mask_test])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_train = dataframe[mask_train]['emotion'].values
    y_test = dataframe[mask_test]['emotion'].values

    return X_train, y_train, X_test, y_test

