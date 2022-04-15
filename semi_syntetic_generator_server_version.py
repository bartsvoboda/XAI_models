from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import DecisionListClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

CLF_NAME = "EBM"

clfs = {
    "CART": DecisionTreeClassifier(random_state=1234),
    "EBM": ExplainableBoostingClassifier(),
    "LR_l2": LogisticRegression(penalty="l2",random_state=1234),
    "GNB": GaussianNB(),
    "LR": LogisticRegression(penalty="none", random_state=1234),
    "DL": DecisionListClassifier(random_state=1234) 
}

datasets = ['breast', 'campus', 'churn', 'climate',
            'compas', 'diabetes', 'german', 'heart',
            'adult', 'student', 'bank', 'credit']

import numpy as np
from sklearn.model_selection import StratifiedKFold
n_datasets = len(datasets)
n_splits = 10

skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)
pred_labels = np.zeros((len(clfs)+1, n_datasets, n_splits))

from sklearn.base import clone 
from sklearn import metrics
import pandas as pd

import helper
import importlib
importlib.reload(helper)
from sklearn.pipeline import make_pipeline


preds_datasets = []

for data_id, dataset in enumerate(datasets):
    X=pd.read_csv(f"datasets/cleaned/{dataset}_X.csv")
    X = X.drop("Unnamed: 0", axis=1)
    y = pd.read_csv(f"datasets/cleaned/{dataset}_y.csv")
    y = y.drop("Unnamed: 0", axis=1)

    features_types_df = pd.read_csv(f"datasets/cleaned/datatypes/{dataset}.csv")

    feature_inidices = list(map(int, list(features_types_df)))
    features_names = list(features_types_df.T[0])
    features_types = list(map(int, list(features_types_df.T[1])))

    preprocess = helper.select_preprocessing_for_many_feat(feature_inidices, features_types, features_names)

    preds_folds = []

    for fold_id, (train, test) in enumerate(skf.split(X, y)):
        clf = clone(clfs[CLF_NAME])

        clf_pipeline = make_pipeline(
            preprocess,
            clf
        )

        print(X.iloc[train])
        print(y.iloc[train])
                
        clf_pipeline.fit(X.iloc[train], y.iloc[train])
        y_preds = clf_pipeline.predict(X)
        preds_folds.append(y_preds)
        
    print(dataset)
    
    preds_datasets.append(preds_folds)


for data_id in range(n_datasets):
    final_labels = pd.DataFrame(np.round(np.mean(preds_datasets[data_id], axis=0)), dtype=int)
    final_labels.to_csv(f"./semi-syntetic_dataset/{CLF_NAME}/{datasets[data_id]}_y.csv", index=False)
