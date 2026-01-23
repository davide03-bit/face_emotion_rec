import numpy as np
from sklearn.preprocessing import StandardScaler

def pixels_to_matrix(dataframe):
    pixels = dataframe['pixels'].apply(lambda x: np.fromstring(x, sep=' '))
    X = np.vstack(pixels)
    return X

def data_split(dataframe):
    data_training= dataframe[(dataframe['Usage'] == 'Training') | (dataframe['Usage'] == 'PrivateTest')]

    X_training = pixels_to_matrix(data_training)
    y_training = data_training['emotion'].values

    dataframe_test = dataframe[dataframe['Usage'] == 'PublicTest']
    X_test = pixels_to_matrix(dataframe_test)
    y_test = dataframe_test['emotion'].values

    std = StandardScaler()
    X_training= std.fit_transform(X_training)
    X_test= std.transform(X_test)
    
    is_train = (data_training['Usage'] == 'Training').values
    is_val = (data_training['Usage'] == 'PrivateTest').values

    X_train = X_training[is_train]
    y_train = y_training[is_train]
    
    X_validation = X_training[is_val]
    y_validation = y_training[is_val]


    return X_train, y_train, X_validation, y_validation, X_test, y_test

