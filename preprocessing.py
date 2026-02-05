import numpy as np

np.random.seed(42)

def pixels_to_matrix(dataframe):
    pixels = dataframe['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    X = np.vstack(pixels)
    return X

def data_split(dataframe):
    mask_train = (dataframe['Usage'] == 'Training') | (dataframe['Usage'] == 'PrivateTest')
    mask_test = dataframe['Usage'] == 'PublicTest'

    X_train = pixels_to_matrix(dataframe[mask_train])
    X_test = pixels_to_matrix(dataframe[mask_test])

    y_train = dataframe[mask_train]['emotion'].values
    y_test = dataframe[mask_test]['emotion'].values

    return X_train, y_train, X_test, y_test

