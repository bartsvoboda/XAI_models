# Pipeline preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


    # return preprocesing for num features only
def num_feat_preprocessing(num_names):
    preprocess = make_column_transformer(
        (StandardScaler(), num_names)
    )
    return preprocess

    # return preprocesing for all features
def feat_preprocessing(num_names, cat_names):
    preprocess = make_column_transformer(
        (OneHotEncoder(), cat_names),
        (StandardScaler(), num_names)
    )
    return preprocess

# return preprocesing for cat features only
def cat_feat_preprocessing(cat_names):
    preprocess = make_column_transformer(
        (OneHotEncoder(), cat_names)
    )
    return preprocess
    

def select_preprocessing_for_many_feat(features_indicies, features_types, features_names):
    # return preprocesing for num features only
    cat_feat = []
    num_feat = []

    for feat_index in features_indicies:
        if features_types[feat_index] == 0:
            num_feat.append(features_names[feat_index])
        else:
            cat_feat.append(features_names[feat_index])
    
    #select preprocesing
    if len(cat_feat) == 0 and len(num_feat) != 0:
        preprocess = num_feat_preprocessing(num_feat)
    if len(cat_feat) != 0 and len(num_feat) == 0:
        preprocess = cat_feat_preprocessing(cat_feat)
    else:
        preprocess = feat_preprocessing(num_feat, cat_feat)

    return preprocess




