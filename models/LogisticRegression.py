# %%
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

import Roland_utils.dataloader



def train_logistic_regression_k_fold_x_order_polynomial(
    features: pd.DataFrame, labels: pd.DataFrame, k: int, order: int, random_state: int
):
    """
    Description:
        According to given order, build a logistic regression classifier by using
        a polynomial function of order 'order'.
        And then, use k-fold-cross-validation to generate traning set and test set
        from features.
        And then, using the training set to train the model, and generate k F1 score
        on training set and test set for each fold
        Finally, return the List of the k train f1 score and k test f1 score.

    Return:
     (train_f1_score_list, test_f1_score_list)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    f1_scores_train_all = []
    f1_scores_test_all = []
    model = LogisticRegression(solver="liblinear")
    polynomial_function = PolynomialFeatures(degree=order)
    transformed_feautres = pd.DataFrame(polynomial_function.fit_transform(features))
    for train_index, test_index in kf.split(transformed_feautres):
        X_train, X_test = (
            transformed_feautres.iloc[train_index, :],
            transformed_feautres.iloc[test_index, :],
        )
        y_train, y_test = labels[train_index], labels[test_index]

        model.fit(X_train, y_train)

        y_train_pred_values = model.predict(X_train)
        y_test_pred_values = model.predict(X_test)

        train_f1_score = f1_score(y_train, y_train_pred_values)
        test_f1_score = f1_score(y_test, y_test_pred_values)

        f1_scores_train_all.append(train_f1_score)
        f1_scores_test_all.append(test_f1_score)
    return f1_scores_train_all, f1_scores_test_all
