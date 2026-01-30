import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def Classification(X_train, X_test, y_train, V):

    pipeline= Pipeline(steps=[
        ('PCA', KernelPCA(n_components= V.shape[1])),
        ('SVM', SVC(class_weight='balanced'))
    ])
    param_grid = {
        'PCA__kernel': ['linear', 'rbf', 'sigmoid'],
        'SVM__kernel': ['linear', 'rbf', 'sigmoid'],
        'SVM__C': [0.1, 1, 10, 100],
        'SVM__gamma': [1, 0.1, 0.01, 0.001]}

    kfold= KFold(n_splits= 5, shuffle=True, random_state=42)

    grid = GridSearchCV(pipeline, param_grid, verbose=True, cv=kfold, n_jobs=-1)
    grid.fit(X_train, y_train)
    classifier= grid.best_estimator_
    print(grid.best_params_)

    return classifier.predict(X_test)



def PCA(X, V):
    Z= np.dot(X, V)
    return Z
