from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

#Support vector machines 
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

clfs = {
    "CART": DecisionTreeClassifier(random_state=1234, max_depth=1000),
    "RNF": RandomForestClassifier(random_state=1234),
    "XGB": XGBClassifier(use_label_encoder=False),
    "CAT": CatBoostClassifier(random_state=1234),
    "ADA": AdaBoostClassifier(DecisionTreeClassifier(random_state=1234, max_depth=1000)),
    "BAG": BaggingClassifier(DecisionTreeClassifier(random_state=1234, max_depth=1000))
    # "ADA": AdaBoostClassifier(SVC(random_state=1234, kernel='rbf', probability=True)),
    # "BAG": BaggingClassifier(SVC(random_state=1234, kernel='rbf', probability=True))
}

datasets = ['breast', 'campus', 'churn', 'climate',
            'compas', 'diabetes', 'german', 'heart',
            'adult', 'student', 'bank', 'credit']


from sklearn.metrics import recall_score, precision_score, accuracy_score,f1_score, auc, roc_curve
metrics_dict = {
    "recall": recall_score,
    'precision': precision_score,
    'accuracy': accuracy_score,
    'f1': f1_score,
    'auc': auc
}

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
n_datasets = len(datasets)
n_splits = 10
# repeats 5, splits 2
skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)

scores = np.zeros((len(clfs), n_datasets, n_splits, len(metrics_dict)))
# loss = np.zeros((len(clfs), n_datasets, n_splits))

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
    y = pd.read_csv(f"../datasets/cleaned/{dataset}_y.csv")
    y = y.drop("Unnamed: 0", axis=1)

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
            print(dataset)
            print(clf_name)
            for metric_id, metric in enumerate(metrics_dict):
                if metric_id == 4:
                    fpr, tpr, thresholds = metrics.roc_curve(y.iloc[test], y_preds)
                    scores[clf_id, data_id, fold_id, metric_id] = metrics.auc(fpr, tpr)
                else:
                    scores[clf_id, data_id, fold_id, metric_id] = metrics_dict[metric](y.iloc[test], y_preds)


np.save('./scores', scores)
# np.save('./test_results/auc/auc_losses', loss)