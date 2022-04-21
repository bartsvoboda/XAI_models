import pandas as pd
import numpy as np 

datasets = ['breast', 'campus', 'churn', 'climate',
            'compas', 'diabetes', 'german', 'heart',
            'adult', 'student', 'bank', 'credit']

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
    "LR": LogisticRegression(penalty="none", random_state=1234),
    # "DL": DecisionListClassifier(random_state=1234) 
}

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import log_loss
n_datasets = len(datasets)
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)
subsample = StratifiedShuffleSplit(n_splits = 8, train_size= 0.5, test_size=0.5, random_state=1234)

variances = np.zeros((n_datasets, n_splits))
biases = np.zeros((n_datasets, n_splits))

from sklearn.base import clone 
from sklearn import metrics
import pandas as pd

import helper
import importlib
importlib.reload(helper)
from sklearn.pipeline import make_pipeline

#RUN PARAM
CLF_NAME = "CART"

#CONST PARAM
GEN_NAME = "DL"

for data_id, dataset in enumerate(datasets):
    X=pd.read_csv(f"../datasets/cleaned/{dataset}_X.csv")
    X = X.drop("Unnamed: 0", axis=1)
    y = pd.read_csv(f"../semi-syntetic_dataset/{GEN_NAME}/{dataset}_y.csv")

    features_types_df = pd.read_csv(f"../datasets/cleaned/datatypes/{dataset}.csv")

    feature_inidices = list(map(int, list(features_types_df)))
    features_names = list(features_types_df.T[0])
    features_types = list(map(int, list(features_types_df.T[1])))

    preprocess = helper.select_preprocessing_for_many_feat(feature_inidices, features_types, features_names)

    subset_results=[]
    subset_acc=[]
    ground_true_labels=[]

    for fold_id, (train, test) in enumerate(skf.split(X, y)):
        subset_results=[]
        subset_acc=[]
        ground_true_labels=[]

        X_train = X.iloc[train]
        X_test = X.iloc[test]

        y_train = y.iloc[train]
        y_test = y.iloc[test]

        for sub_id, (train_idx, test_idx) in enumerate(subsample.split(X_train, y_train)):
            X_sub_train, y_sub_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
            
            clf = clone(clfs[CLF_NAME])
            clf_pipeline = make_pipeline(
                preprocess,
                clf
            )

            clf_pipeline.fit(X_sub_train, y_sub_train)
            clf_preds = clf_pipeline.predict(X_test)
            subset_results.append(clf_preds)
            ground_true_labels.append(y_test)

        variance = np.mean(np.var(np.array(subset_results), axis=0))
        avg_test_y_pred = np.mean(np.array(subset_results), axis=0)
        bias = np.mean((avg_test_y_pred - ground_true_labels) ** 2)

        biases[data_id, fold_id] = bias
        variances[data_id, fold_id] = variance

columns_names = ["dataset_name","fold_id", "bias", "variance"]

temp_dfs = []

for i in range(len(datasets)):
    temp_df = pd.DataFrame(columns=columns_names)
    temp_df["fold_id"] = np.arange(10)
    temp_df["dataset_name"] = datasets[i]
    temp_df["bias"]=biases[i]
    temp_df["variance"]=variances[i]
    temp_dfs.append(temp_df)

results = pd.concat(temp_dfs)
results.to_csv(f"./{GEN_NAME}/{CLF_NAME}.csv",index=False)