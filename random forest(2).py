# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:02:33 2024

@author: ezrea
"""

from torch_geometric.datasets import Planetoid, WikiCS
import torch
# Load the dataset
cora = Planetoid(root='/tmp/Cora', name='Cora')
wikics = WikiCS(root='/tmp/WikiCS')

# Access the first graph object
#data = dataset[0]
print(f'Dataset: {wikics}:')
print('======================')
print(f'Number of graphs: {len(wikics)}')
print(f'Number of features: {wikics.num_features}')
print(f'Number of classes: {wikics.num_classes}')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, recall_score,precision_score, f1_score, accuracy_score
import pandas as pd
from typing import Literal
from sklearn.model_selection import KFold

x_cora = cora[0].x
y_cora = cora[0].y

x_wikics = wikics[0].x
y_wikics = wikics[0].y

features = x_wikics
labels = y_wikics
features = pd.DataFrame(features)
labels = pd.Series(labels)
random_state = 1
k = 5

#Use wiki data
# features = x_wikics
# labels = y_wikics
# features = pd.DataFrame(features)
# labels = pd.Series(labels)

# def train_random_forest_k_fold(
#     features: pd.DataFrame,
#     labels: pd.DataFrame,
#     k: int,
#     random_state: int    
# ):
#     """
#     Description:
#         use k-fold-cross-validation to generate traning set and test set from features.
#         And then, using the training set to train the model, and generate different criterion
#         on training set and test set for each fold, and then calculate the average
#         Finally, return the the k criterions average

#     Return:
#         {
#             "accuracy": accuracy,
#             "precision_macro": precision_macro,
#             "recall_marco": recall_marco,
#             "f1_marco": f1_marco,
#             "precision_micro": precision_micro,
#             "recall_micro": recall_micro,
#             "f1_micro": f1_micro,
#         },
#     """
X_train_val, X_test, y_train_val, y_test = train_test_split(
    features.reset_index(drop=True),  # Reset index here
    labels, 
    test_size=0.8, 
    random_state=random_state, 
    stratify=labels
)

y_train_val = np.array(y_train_val)

kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
model = RandomForestClassifier(n_estimators = 100, max_depth = None)

accuracy_list = []
precision_macro_list = []
recall_marco_list = []
f1_marco_list = []
precision_micro_list = []
recall_micro_list = []
f1_micro_list = []

best_model = None
best_f1_score = 0 

for train_index, val_index in kf.split(X_train_val):
    X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
    y_train, y_val = y_train_val[train_index], y_train_val[val_index]
    
    model.fit(X_train, y_train)
    y_val_pred_values = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_val_pred_values)
    accuracy_list.append(accuracy)
    
    precision_macro = precision_score(
                y_true=y_val,
                y_pred=y_val_pred_values,
                average="macro",
                zero_division=0,
            )
    precision_macro_list.append(precision_macro)
    
    precision_micro = precision_score(
                y_true=y_val,
                y_pred=y_val_pred_values,
                average="micro",
                zero_division=0,
            )
    precision_micro_list.append(precision_micro)
    recall_marco = recall_score(y_true=y_val, y_pred=y_val_pred_values, average="macro")
    recall_marco_list.append(recall_marco)
    recall_micro = recall_score(y_true=y_val, y_pred=y_val_pred_values, average="micro")
    recall_micro_list.append(recall_micro)
    f1_marco = f1_score(y_true=y_val, y_pred=y_val_pred_values, average="macro")
    f1_marco_list.append(f1_marco)
    f1_micro = f1_score(y_true=y_val, y_pred=y_val_pred_values, average="micro")
    f1_micro_list.append(f1_micro)
    # 如果当前模型比之前的最佳模型表现好，则更新最佳模型
    if f1_marco > best_f1_score:
        best_f1_score = f1_marco
        best_model = model

avg_accuracy = np.mean(accuracy_list)
avg_precision_macro = np.mean(precision_macro_list)
avg_recall_macro = np.mean(recall_marco_list)
avg_f1_macro = np.mean(f1_marco_list)
avg_precision_micro = np.mean(precision_micro_list)
avg_recall_micro = np.mean(recall_micro_list)
avg_f1_micro = np.mean(f1_micro_list)


#Test set performance
y_test_pred = best_model.predict(X_test)

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision_macro = precision_score(y_test, y_test_pred, average='macro')
test_recall_macro = recall_score(y_test, y_test_pred, average='macro')
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')

print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision (Macro): {test_precision_macro}')
print(f'Test Recall (Macro): {test_recall_macro}')
print(f'Test F1 Score (Macro): {test_f1_macro}')

test_precision_micro = precision_score(y_test, y_test_pred, average='micro')
test_recall_micro = recall_score(y_test, y_test_pred, average='micro')
test_f1_micro = f1_score(y_test, y_test_pred, average='micro')


    # return (
    #     {
    #         "accuracy": np.mean(accuracy_list),
    #         "precision_macro": np.mean(precision_macro_list),
    #         "recall_marco": np.mean(recall_marco_list),
    #         "f1_marco": np.mean(f1_marco_list),
    #         "precision_micro": np.mean(precision_micro_list),
    #         "recall_micro": np.mean(recall_micro_list),
    #         "f1_micro": np.mean(f1_micro_list),
    #     },
    # )



# train_random_forest_k_fold(
#     features=features,
#     labels=labels,
#     k=5,
#     random_state=1
# )
# Result of cora
# ({'accuracy': 0.8744432545989046,
#   'precision_macro': 0.8883802614998636,
#   'recall_marco': 0.849813351384898,
#   'f1_marco': 0.8625146496840645,
#   'precision_micro': 0.8744432545989046,
#   'recall_micro': 0.8744432545989046,
#   'f1_micro': 0.8744432545989046},)


# train_random_forest_k_fold(
#     features=features,
#     labels=labels,
#     k=5,
#     random_state=1
# )
# 输出模型参数
# Result of wikics
# ({'accuracy': 0.8741985673446587,
#   'precision_macro': 0.8901024576902203,
#   'recall_marco': 0.8306450155090659,
#   'f1_marco': 0.8511605341929671,
#   'precision_micro': 0.8741985673446587,
#   'recall_micro': 0.8741985673446587,
#   'f1_micro': 0.8741985673446588},)