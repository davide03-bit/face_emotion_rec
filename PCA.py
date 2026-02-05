import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


def Classification(X_train, y_train, variance_threshold):

    pipeline= Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('PCA', PCA(n_components=variance_threshold, svd_solver='full', random_state=81)),
        ('SVM', SVC(class_weight='balanced', random_state=42))
    ])

    param_grid = [
        {
            'SVM__kernel': ['linear'],
            'SVM__C': [0.1, 1, 10, 100]
        },
        {
            'SVM__kernel': ['rbf', 'sigmoid'],
            'SVM__C': [0.1, 1, 10, 100],
            'SVM__gamma': [1, 0.1, 0.01, 0.001]
        }
    ]

    kfold = KFold(n_splits= 5, shuffle=True, random_state=42)

    grid = GridSearchCV(pipeline, param_grid, verbose=True, cv=kfold, n_jobs=-1, scoring='f1_weighted')

    grid.fit(X_train, y_train)
    classifier = grid.best_estimator_

    print(f"Migliori iperparametri: {grid.best_params_}")
    print(f"Numero di componenti principali selezionate: {classifier.named_steps['PCA'].n_components_}")

    return classifier

def Classification2(X_train, y_train):

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('SVM', SVC(class_weight='balanced', random_state=42))
    ])

    param_grid = [
    {
        'SVM__kernel': ['linear'],
        'SVM__C': [0.1, 1, 10, 100]
    },
    {
        'SVM__kernel': ['rbf', 'sigmoid'],
        'SVM__C': [0.1, 1, 10, 100],
        'SVM__gamma': [1, 0.1, 0.01, 0.001]
    }
    ]

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(pipeline, param_grid, verbose=True, cv=kfold, n_jobs=-1, scoring='f1_weighted')

    grid.fit(X_train, y_train)
    classifier = grid.best_estimator_

    print(f"Migliori iperparametri: {grid.best_params_}")
    return classifier
