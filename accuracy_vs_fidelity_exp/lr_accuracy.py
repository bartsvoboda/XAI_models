from sklearn.tree import DecisionTreeClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from interpret.glassbox import DecisionListClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


clfs = {
    "CART": DecisionTreeClassifier(random_state=1234),
    "EBM": ExplainableBoostingClassifier(),
    "LR_l2": LogisticRegression(penalty="l2",random_state=1234),
    "GNB": GaussianNB(),
    # "LR": LogisticRegression(penalty="none", random_state=1234) 
}

generator_name = "LR"

datasets = ['breast', 'campus', 'churn', 'climate',
            'compas', 'diabetes', 'german', 'heart',
            'adult', 'student', 'bank', 'credit']

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
n_datasets = len(datasets)
n_splits = 10
# repeats 5, splits 2
skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)

auc_scores = np.zeros((len(clfs)+1, n_datasets, n_splits))
loss = np.zeros((len(clfs)+1, n_datasets, n_splits))

from sklearn.base import clone 
from sklearn import metrics
import pandas as pd

import helper
import importlib
importlib.reload(helper)
from sklearn.pipeline import make_pipeline

for data_id, dataset in enumerate(datasets):
    X=pd.read_csv(f"../datasets/cleaned/{dataset}_X.csv")
    X = X.drop("Unnamed: 0", axis=1)
    y = pd.read_csv(f"../semi-syntetic_dataset/{generator_name}/{dataset}_y.csv")

    features_types_df = pd.read_csv(f"../datasets/cleaned/datatypes/{dataset}.csv")

    feature_inidices = list(map(int, list(features_types_df)))
    features_names = list(features_types_df.T[0])
    features_types = list(map(int, list(features_types_df.T[1])))

    preprocess = helper.select_preprocessing_for_many_feat(feature_inidices, features_types, features_names)

    for fold_id, (train, test) in enumerate(skf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf_pipeline = make_pipeline(
                preprocess,
                clf
            )
                
            clf_pipeline.fit(X.iloc[train], y.iloc[train])
            y_preds = clf_pipeline.predict(X.iloc[test])
            fpr, tpr, thresholds = metrics.roc_curve(y.iloc[test], y_preds)
            auc_scores[clf_id, data_id, fold_id] = metrics.auc(fpr, tpr)
            loss[clf_id, data_id, fold_id] = log_loss(y.iloc[test], y_preds)

from sklearn.base import clone 
from sklearn import metrics
import pandas as pd

import helper
import importlib
importlib.reload(helper)
from sklearn.pipeline import make_pipeline

for data_id, dataset in enumerate(datasets):
    X=pd.read_csv(f"../datasets/cleaned/{dataset}_X.csv")
    X = X.drop("Unnamed: 0", axis=1)
    y = pd.read_csv(f"../semi-syntetic_dataset/{generator_name}/{dataset}_y.csv")

    features_types_df = pd.read_csv(f"../datasets/cleaned/datatypes/{dataset}.csv")

    feature_inidices = list(map(int, list(features_types_df)))
    features_names = list(features_types_df.T[0])
    features_types = list(map(int, list(features_types_df.T[1])))

    preprocess = helper.select_preprocessing_for_many_feat(feature_inidices, features_types, features_names)

    for fold_id, (train, test) in enumerate(skf.split(X, y)):

        clf_pipeline = make_pipeline(
                preprocess,
                DecisionListClassifier(random_state=1234)
            )
                
        clf_pipeline.fit(X.iloc[train], y.iloc[train])
        y_preds = clf_pipeline.predict(X.iloc[test])
        fpr, tpr, thresholds = metrics.roc_curve(y.iloc[test], y_preds)
        auc_scores[4, data_id, fold_id] = metrics.auc(fpr, tpr)
        loss[4, data_id, fold_id] = log_loss(y.iloc[test], y_preds)

np.save(f'./auc_results_{generator_name}', auc_scores)
np.save(f'./auc_losses_{generator_name}', loss)
        