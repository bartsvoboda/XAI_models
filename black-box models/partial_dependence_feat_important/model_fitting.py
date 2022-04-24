import dalex as dx
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

#Support vector machines 
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier


clfs = {
    "CART": DecisionTreeClassifier(random_state=1234),
    "RNF": RandomForestClassifier(random_state=1234),
    "XGB": XGBClassifier(use_label_encoder=False),
    "CAT": CatBoostClassifier(random_state=1234),
    "ADA": AdaBoostClassifier(SVC(random_state=1234, kernel='rbf', probability=True)),
    "BAG": BaggingClassifier(SVC(random_state=1234, kernel='rbf', probability=True))
}

dataset = 'adult'

import worstcase_helper
import importlib
importlib.reload(worstcase_helper)

preprocess, X, y = worstcase_helper.load_dataset_with_preprocess(dataset)

from sklearn.pipeline import make_pipeline
def make_pipeline_clf(clf_name):
    clf = make_pipeline(
        preprocess,
        clfs[clf_name]
    )
    return clf

# clf_cart = make_pipeline_clf("CART")
# clf_cart.fit(X, y)

clf_rnf = make_pipeline_clf("RNF")
clf_rnf.fit(X, y)

# clf_xgb = make_pipeline_clf("XGB")
# clf_xgb.fit(X, y)

# clf_cat = make_pipeline_clf("CAT")
# clf_cat.fit(X, y)

clf_ada = make_pipeline_clf("ADA")
clf_ada.fit(X, y)

clf_bag = make_pipeline_clf("BAG")
clf_bag.fit(X, y)

import pickle
dataset = "adult"
pickle.dump(clf_rnf, open(f"./models/{dataset}_rnf", 'wb'))
pickle.dump(clf_ada, open(f"./models/{dataset}_ada", 'wb'))
pickle.dump(clf_bag, open(f"./models/{dataset}_bag", 'wb'))