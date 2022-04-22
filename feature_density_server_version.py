import pandas as pd
import numpy as np 

datasets = ['breast', 'campus', 'churn', 'climate',
            'compas', 'diabetes', 'german', 'heart',
            'stroke', 'student', 'water', 'credit']

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
    "DL": DecisionListClassifier(random_state=1234) 
}

#Methods
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
# Pipeline preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss

def get_output_col_name(output_queue, col_names):
  output_features_names = pd.Series(dtype="string")
  for i in range(len(output_queue)):
    output_features_names = pd.concat([output_features_names, pd.Series([col_names[output_queue].iloc[i]])])
  return output_features_names

# return preprocesing for num features only
def num_feat_preprocessing(num_names):
  preprocess = make_column_transformer(
      (StandardScaler(), num_names)
  )
  return preprocess

# return preprocesing for cat features only
def cat_feat_preprocessing(cat_names):
  preprocess = make_column_transformer(
      (OneHotEncoder(), cat_names)
  )
  return preprocess

# return preprocesing for all features
def feat_preprocessing(num_names, cat_names):
  preprocess = make_column_transformer(
      (OneHotEncoder(), cat_names),
      (StandardScaler(), num_names)
  )
  return preprocess

def select_preprocessing_for_single_feat(init_index, col_names, col_types):
  #tested
  cat_feat = []
  num_feat = []

  print(f"pre: {col_names}\n{col_types}")
  print(f"pre init: {init_index}")

  if col_types[int(init_index)] == 0:
    num_feat.append(col_names[int(init_index)])
    #run StandardScaler function
    preprocess = num_feat_preprocessing(num_feat)
  else:
    cat_feat.append(col_names[int(init_index)])
    preprocess = cat_feat_preprocessing(cat_feat)
  return preprocess

def select_preprocessing_for_many_feat(output_col_names, col_names, col_types):
  cat_feat = []
  num_feat = []

  for feat_index in output_col_names:
    if col_types[feat_index] == 0:
      num_feat.append(col_names[feat_index])
    else:
      cat_feat.append(col_names[feat_index])
  
  print(cat_feat)
  print(num_feat)
  
  #select preprocesing
  if len(cat_feat) == 0 and len(num_feat) != 0:
    preprocess = num_feat_preprocessing(num_feat)
    print("Jestem tu!!!")
  if len(cat_feat) != 0 and len(num_feat) == 0:
    preprocess = cat_feat_preprocessing(cat_feat)
  else:
    preprocess = feat_preprocessing(num_feat, cat_feat)
  return preprocess

def create_data_frame_for_feat(output_col_names, dataset_df):
  # if len(output_col_names) == 1:
  #   return pd.DataFrame(dataset_df[output_col_names], columns=[output_col_names])
  # else:
    return dataset_df[output_col_names]

def calculate_loss_for_single_feat(X_df, y_lab, init_index, train_indices, test_indices, f_names, f_types):
  X = X_df
  y = y_lab

  print(f"\n funkcja: {f_names} \n {f_types}")
  print(init_index)

  preprocess = select_preprocessing_for_single_feat(init_index=int(init_index),
                                                  col_names=f_names,
                                                  col_types=f_types)

    # Split beetwen three dataset (test, train, val)
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1337, shuffle=True)
  # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1337)

  clf = make_pipeline(
      preprocess,
      ExplainableBoostingClassifier()
  )

  clf.fit(X.iloc[train_indices], y.iloc[train_indices])

  #Prediction
  y_preds = clf.predict(X.iloc[test_indices])

  #Calculate logloss
  # p = np.clip(y_preds, 1e-12, 1. - 1e-12)
  # result= np.mean(y_test * -np.log(p) + (1. - y_test) * (-np.log(1. - p)))
  result = log_loss(y.iloc[test_indices], y_preds)

  return(result, X.columns[0])

def calculate_loss_for_multi_feat(X_df, y_lab, output_with_to_pred_feat, train_indices, test_indices, f_names, f_types):
  print(X_df)
  print(y_lab)
  X = X_df
  y = y_lab

  preprocess = select_preprocessing_for_many_feat(output_col_names=output_with_to_pred_feat,
                                                  col_names=f_names,
                                                  col_types=f_types)
  print(preprocess)
    # Split beetwen three dataset (test, train, val)
  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1337, shuffle=True)
  # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1337)

  clf = make_pipeline(
      preprocess,
      ExplainableBoostingClassifier()
  )

  clf.fit(X.iloc[train_indices], y.iloc[train_indices])

  #Prediction
  y_preds = clf.predict(X.iloc[test_indices])

  #Calculate logloss
  # p = np.clip(y_preds, 1e-12, 1. - 1e-12)
  # result= np.mean(y_test * -np.log(p) + (1. - y_test) * (-np.log(1. - p)))
  result = log_loss(y.iloc[test_indices], y_preds)

  return(result, X.columns[0])

def concat_data_indices(output_queue, input_queue):
  temp_list = list(output_queue)
  indices_list = []
  for index_input in input_queue:
    temp_list.append(index_input)
    indices_list.append(temp_list)
    temp_list = list(output_queue)
  return indices_list

def calculate_and_save_losses(fold_id, clf_name, dataset_name, train_idx, test_idx):
    features_types_df = pd.read_csv(f"datasets/cleaned/datatypes/{dataset_name}.csv")

    feature_inidices = list(map(int, list(features_types_df)))
    features_names = pd.Series(list(features_types_df.T[0]))
    features_types = pd.Series(list(map(int, list(features_types_df.T[1]))))
    print(f"\n{features_types}")

    #define data containers for features
    #input_queue: indices of features to be check
    #output_queue: indices of checked features (order is coresponding to given loss)
    #data_losses: data container for each step losses storing
    input_queue = pd.Series(feature_inidices, dtype=int)
    output_queue = pd.Series([], dtype=int)
    run_losses = pd.Series([], dtype=float)

    initial_index = 0
    test_df = create_data_frame_for_feat(get_output_col_name([initial_index], features_names), X)
    result, name = calculate_loss_for_single_feat(test_df, y, initial_index, train_idx, test_idx, features_names, features_types)

    initial_error = result
    initial_name = name

    losses_vector = np.zeros(len(input_queue))
    for index in feature_inidices:
        test_df = create_data_frame_for_feat(get_output_col_name([index], features_names), X)
        result, name = calculate_loss_for_single_feat(test_df, y, index,train_idx, test_idx, features_names, features_types)
        losses_vector[index] = result
        print(name)

    run_losses[0] = losses_vector
    # get index of smallest loses feature
    feature_selected_index = input_queue.iloc[run_losses[0].argmin()]
    #pop index from input queue
    input_queue.pop(feature_selected_index)
    #add selected index to output_queue
    output_queue = pd.concat([output_queue, pd.Series(feature_selected_index)])

    for i in range(len(input_queue)):
        losses_vector = np.zeros(len(input_queue))
        lista_test = concat_data_indices(output_queue, input_queue)

        for j in range(len(input_queue)):
            test_df = create_data_frame_for_feat(get_output_col_name(list(lista_test[j]), features_names), X)
            result, name = calculate_loss_for_multi_feat(test_df, y, list(lista_test[j]), train_idx, test_idx, features_names, features_types)
            losses_vector[j] = result
    
        run_losses[i+1] = losses_vector
        # get index of smallest loses feature
        feature_selected_index = input_queue.iloc[run_losses[i+1].argmin()]
        input_queue.pop(feature_selected_index)
        #add selected index to output_queue
        output_queue = pd.concat([output_queue, pd.Series(feature_selected_index)])

    sorted_results = np.zeros(len(run_losses))
    # sorted_results[0] = initial_error
    for i in range(len(run_losses)):
        print(run_losses[i].min())
        sorted_results[i] = run_losses[i].min()

    final_results = []
    initial_result = [initial_index, "initial_error", initial_error]
    final_results.append(initial_result)

    for i in range(len(output_queue)):
        temp_result = [output_queue.iloc[i],features_names[output_queue].iloc[i], sorted_results[i]]
        final_results.append(temp_result)
    pd.DataFrame(final_results).to_csv(index=False, path_or_buf=f"./test_results/feature_density/{clf_name}_{dataset_name}_{fold_id}.csv")


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
n_datasets = len(datasets)
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, random_state=1234, shuffle=True)

# for dataset in datasets:
    #import dataset
dataset = "german"
X = pd.read_csv(f"./datasets/cleaned/{dataset}_X.csv")
X = X.drop("Unnamed: 0", axis=1)
y = pd.read_csv(f"./datasets/cleaned/{dataset}_y.csv")
y = y.drop("Unnamed: 0", axis=1)
X.head()

for fold_id, (train, test) in enumerate(skf.split(X, y)):
  calculate_and_save_losses(fold_id, "GNB", dataset, train_idx=train, test_idx=test)
    