# Pipeline preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


    # return preprocesing for num features only
def num_feat_preprocessing(num_names):
    preprocess = make_column_transformer(
        (StandardScaler(), num_names)
        # (MinMaxScaler(), num_names)

    )
    return preprocess

    # return preprocesing for all features
def feat_preprocessing(num_names, cat_names):
    preprocess = make_column_transformer(
        (OneHotEncoder(), cat_names),
        (StandardScaler(), num_names),
        # (MinMaxScaler(), num_names)

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

def get_output_col_name(output_queue, col_names):
  output_features_names = pd.Series(dtype="string")
  for i in range(len(output_queue)):
    output_features_names = pd.concat([output_features_names, pd.Series([col_names[output_queue].iloc[i]])])
  return output_features_names


def select_preprocessing_for_single_feat(init_index, col_names, col_types):
  #tested
  cat_feat = []
  num_feat = []

  if col_types[int(init_index)] == 0:
    num_feat.append(col_names[int(init_index)])
    #run StandardScaler function
    preprocess = num_feat_preprocessing(num_feat)
  else:
    cat_feat.append(col_names[int(init_index)])
    preprocess = cat_feat_preprocessing(cat_feat)
  return preprocess

def create_data_frame_for_feat(output_col_names, dataset_df):
# if len(output_col_names) == 1:
# return pd.DataFrame(dataset_df[output_col_names], columns=[output_col_names])
# else:
    return dataset_df[output_col_names]


def calculate_loss_for_single_feat(X_df, y_lab, init_index, dataset_col_names, dataset_col_types, model):
  X = X_df
  y = y_lab

  preprocess = select_preprocessing_for_single_feat(init_index=int(init_index),
                                                    col_names=dataset_col_names,
                                                    col_types=dataset_col_types)

    # Split beetwen three dataset (test, train, val)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1337, shuffle=True)
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1337)

  adult_ebm = make_pipeline(
      preprocess,
      model
  )

  adult_ebm.fit(X_train, y_train)

  #Prediction
  y_preds = adult_ebm.predict(X_test)

  #Calculate logloss
  p = np.clip(y_preds, 1e-12, 1. - 1e-12)
  result= np.mean(y_test * -np.log(p) + (1. - y_test) * (-np.log(1. - p)))

  return(result, X.columns[0])


def calculate_loss_for_multi_feat(X_df, y_lab, output_with_to_pred_feat, dataset_col_names, dataset_col_types, model):
  print(X_df)
  print(y_lab)
  X = X_df
  y = y_lab

  preprocess = select_preprocessing_for_many_feat(features_indicies=output_with_to_pred_feat,
                                                  features_types=dataset_col_types,
                                                  features_names=dataset_col_names)
  print(preprocess)
    # Split beetwen three dataset (test, train, val)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1337, shuffle=True)
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1337)

  adult_ebm = make_pipeline(
      preprocess,
      model
  )

  adult_ebm.fit(X_train, y_train)

  #Prediction
  y_preds = adult_ebm.predict(X_test)

  #Calculate logloss
  p = np.clip(y_preds, 1e-12, 1. - 1e-12)
  result= np.mean(y_test * -np.log(p) + (1. - y_test) * (-np.log(1. - p)))

  return(result, X.columns[0])