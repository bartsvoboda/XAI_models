def load_dataset_with_preprocess(clf_name, dataset):
    import pandas as pd
    import helper
    # import importlib
    # importlib.reload(..helper)

    X=pd.read_csv(f"./datasets/{dataset}_X.csv")
    X = X.drop("Unnamed: 0", axis=1)
    y = pd.read_csv(f"../semi-syntetic_dataset/{clf_name}/{dataset}_y.csv")

    features_types_df = pd.read_csv(f"../datasets/cleaned/datatypes/{dataset}.csv")

    feature_inidices = list(map(int, list(features_types_df)))
    features_names = list(features_types_df.T[0])
    features_types = list(map(int, list(features_types_df.T[1])))

    preprocess = helper.select_preprocessing_for_many_feat(feature_inidices, features_types, features_names)
    return preprocess, X, y 