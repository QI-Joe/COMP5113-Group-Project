# %%
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from typing import Literal

import Roland_utils.dataloader


def train_logistic_regression_k_fold_multi_class(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    k: int,
    random_state: int,
    multi_class: Literal["auto", "ovr", "multinomial"] = "auto",
    solver: Literal[
        "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
    ] = "lbfgs",
):
    """
    Description:
        use k-fold-cross-validation to generate traning set and test set from features.
        And then, using the training set to train the model, and generate different criterion
        on training set and test set for each fold, and then calculate the average
        Finally, return the the k criterions average

    Return:
        {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_marco": recall_marco,
            "f1_marco": f1_marco,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
        },
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    model = LogisticRegression(multi_class=multi_class, solver=solver)
    accuracy_list = []
    precision_macro_list = []
    recall_marco_list = []
    f1_marco_list = []
    precision_micro_list = []
    recall_micro_list = []
    f1_micro_list = []
    for train_index, test_index in kf.split(features):
        X_train, X_test = (
            features.iloc[train_index, :],
            features.iloc[test_index, :],
        )
        y_train, y_test = labels[train_index], labels[test_index]

        model.fit(X_train, y_train)

        y_train_pred_values = model.predict(X_train)
        y_test_pred_values = model.predict(X_test)

        accuracy = np.mean(
            [
                accuracy_score(y_train, y_train_pred_values),
                accuracy_score(y_test, y_test_pred_values),
            ]
        )
        accuracy_list.append(accuracy)
        precision_macro = np.mean(
            [
                precision_score(
                    y_true=y_train,
                    y_pred=y_train_pred_values,
                    average="macro",
                    zero_division=0,
                ),
                precision_score(
                    y_true=y_test,
                    y_pred=y_test_pred_values,
                    average="macro",
                    zero_division=0,
                ),
            ]
        )
        precision_macro_list.append(precision_macro)
        precision_micro = np.mean(
            [
                precision_score(
                    y_true=y_train,
                    y_pred=y_train_pred_values,
                    average="micro",
                    zero_division=0,
                ),
                precision_score(
                    y_true=y_test,
                    y_pred=y_test_pred_values,
                    average="micro",
                    zero_division=0,
                ),
            ]
        )
        precision_micro_list.append(precision_micro)
        recall_marco = np.mean(
            [
                recall_score(
                    y_true=y_train, y_pred=y_train_pred_values, average="macro"
                ),
                recall_score(y_true=y_test, y_pred=y_test_pred_values, average="macro"),
            ]
        )
        recall_marco_list.append(recall_marco)
        recall_micro = np.mean(
            [
                recall_score(
                    y_true=y_train, y_pred=y_train_pred_values, average="micro"
                ),
                recall_score(y_true=y_test, y_pred=y_test_pred_values, average="micro"),
            ]
        )
        recall_micro_list.append(recall_micro)
        f1_marco = np.mean(
            [
                f1_score(y_true=y_train, y_pred=y_train_pred_values, average="macro"),
                f1_score(y_true=y_test, y_pred=y_test_pred_values, average="macro"),
            ]
        )
        f1_marco_list.append(f1_marco)
        f1_micro = np.mean(
            [
                f1_score(y_true=y_train, y_pred=y_train_pred_values, average="micro"),
                f1_score(y_true=y_test, y_pred=y_test_pred_values, average="micro"),
            ]
        )
        f1_micro_list.append(f1_micro)

    return (
        {
            "accuracy": np.mean(accuracy_list),
            "precision_macro": np.mean(precision_macro_list),
            "recall_marco": np.mean(recall_marco_list),
            "f1_marco": np.mean(f1_marco_list),
            "precision_micro": np.mean(precision_micro_list),
            "recall_micro": np.mean(recall_micro_list),
            "f1_micro": np.mean(f1_micro_list),
        },
    )


# %%
datasets = Roland_utils.dataloader.get_dataset(path="./datasets", name="WikiCS")
features = datasets[0].x
labels = datasets[0].y
features = pd.DataFrame(features)
labels = pd.Series(labels)
train_logistic_regression_k_fold_multi_class(
    features=features,
    labels=labels,
    k=5,
    random_state=1,
    multi_class="multinomial",
    solver="newton-cg",
)

# %%
