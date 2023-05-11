from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, \
    precision_score, recall_score

import numpy as np
import pandas as pd
import os

def get_measures_values(y_test, y_pred):
    acc_ = accuracy_score(y_test, y_pred)
    recall_ = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    return acc_, recall_, f1, precision


def save_measures(acc, recall, f1, precision, model_name):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    save = os.path.join(results_dir, f'scores.csv')

    dict_values = {'Accuracy': acc, 'Recall': recall, 'f1-score': f1, 'Precision': precision}

    if os.path.exists(save):
        measures = pd.read_csv(save, index_col=0)
        row = pd.Series(dict_values, name = model_name)
        measures = measures.append(row)
    else: 
        measures = pd.DataFrame(
            dict_values,
            index=[model_name]
            )
    measures = measures.rename_axis("Model names", axis="rows")
    measures.to_csv(save)



def create_data_pipelines(numerical_cols, cols_to_encode):
    cat_data_steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='No info')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
         ]

    categorical_pipeline = Pipeline(steps=cat_data_steps)

    num_data_steps = [
        ('imputer', SimpleImputer(strategy='mean'))
    ]

    num_pipeline = Pipeline(num_data_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical_transformer', num_pipeline, numerical_cols),
            ('categorical_transformer', categorical_pipeline, cols_to_encode)
        ]
    )

    return preprocessor


def separate_data(data, target_column, size=0.25):
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]

    return train_test_split(X, y, test_size=size, random_state=42)


def encoding_data(y_train, y_test):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)
    y_test = encoder.transform(y_test)

    return y_train, y_test


def get_params(model_name):
    depths = [None]
    
    params = dict()
    if model_name == 'Decision Tree':
        depths.extend(np.arange(1, 9, 1, dtype=int).tolist())
        params = {
                    'model__max_depth': depths,
                    'model__min_samples_split': np.arange(2, 6, 1, dtype=int),
                    'model__min_samples_leaf': np.arange(1, 6, 1, dtype=int),
                }
    elif model_name == 'Random Forest':
        params = {
                'model__n_estimators': [50, 100, 200, 300],
                'model__max_depth': [None, 3, 5, 7],
                'model__min_samples_split': [2, 3, 5],
                'model__min_samples_leaf': [1, 2, 4]
            }
        
    elif model_name == 'Gradient Boosting':
        depths.extend(np.arange(1, 5, 1, dtype=int).tolist())
        params = {
                        'model__max_depth': depths,
                        'model__min_samples_split': np.arange(0.01, 0.1, 0.01, dtype=float),
                        'model__learning_rate': np.arange(0.01, 0.1, 0.01, dtype=float)
                    }   
    elif model_name == 'Hist Gradient Boosting':
        depths.extend(np.arange(1, 8, 1, dtype=int).tolist())
        params = {
                    'model__max_depth': depths,
                    'model__max_leaf_nodes': np.arange(10, 100, 10, dtype=int),
                    'model__learning_rate': np.arange(0.01, 0.1, 0.01, dtype=float),
                    'model__max_iter': np.arange(50, 300, 50, dtype=int),
                }
    elif model_name == 'SVC':
        params = {
            'model__gamma': [1, 0.1, 0.01, 0.001],
            'model__kernel': ['linear', 'rbf', 'sigmoid']
        } 

    return params       


def grid_search_pipe(model_clf, model_name, x_train, y_train, preprocessor,
                     measures, model_params=None
                     ):

    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', model_clf)
                            ])

    if model_params is None:
        model_params = get_params(model_name)

    grid_search = GridSearchCV(model_pipeline, model_params,
                               cv=3, scoring=measures, refit='f1'
                               )

    grid_search.fit(x_train, y_train)

    # print(f'Best estimator: {grid_search.best_estimator_}')
    print(f'Best score: {grid_search.best_score_:.2%}')
    print(f'Best parameter combination = {grid_search.best_params_}')
    best_model = grid_search.best_estimator_

    return best_model