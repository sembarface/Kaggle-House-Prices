import os
import time
import random

import numpy as np
import pandas as pd

import optuna

import joblib

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold

from sklearn.metrics import root_mean_squared_error

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import torch

from preprocessing import Preprocessor

from DNN import DNN

def set_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_model(X_train, y_train, X_val, y_val, model_name, fold_index, config, params=None, save_model=True):
    """Train and evaluate a model on one fold and optionally save it."""
    
    print(f"Training {model_name}!")
    
    if params is None:
        params = config.model_parameters[model_name]
    
    
    # Fit preprocessing on each training fold to prevent data leakage
    preprocessor = Preprocessor(model_name, config)
    X_train = preprocessor.fit_transform(X_train, y_train)
    X_val = preprocessor.transform(X_val)
    
    
    # Linear Regression
    if model_name == 'linreg':
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        
    # Lasso
    elif model_name == 'lasso':
        model = Lasso(**params)
        model.fit(X_train, y_train)
    
    # Ridge
    elif model_name == 'ridge':
        model = Ridge(**params)
        model.fit(X_train, y_train)

    # ElasticNet
    elif model_name == 'elasticnet':
        model = ElasticNet(**params)
        model.fit(X_train, y_train)
    
    # KNN
    elif model_name == 'KNN':
        model = KNeighborsRegressor(**params)
        model.fit(X_train, y_train)
    
    # Decision Tree
    elif model_name == 'decision_tree':
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
    
    # Random Forest
    elif model_name == 'random_forest':
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
    
    # CatBoost Regressor
    elif model_name == 'catboost':
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=preprocessor.cat_cols_)
        
    # XGBoost Regressor
    elif model_name == 'xgboost':
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
        
    # LightGBM Regressor
    elif model_name == 'lgbm':
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_X=(X_train, X_val), eval_y=(y_train, y_val), 
                    eval_names=['train', 'val'], eval_metric='rmse')
        
    # Deep Neural Network   
    elif model_name == 'DNN':
        model = DNN(**params)
        model.fit(X_train, y_train, X_val, y_val)
    
    # Stacking with KNN, Decision Tree and Linear Regression
    elif model_name == 'stacking_knn_tree_linreg':
        
        estimators = [
            ('knn', KNeighborsRegressor(**config.model_parameters['KNN'])),
            ('tree', DecisionTreeRegressor(**config.model_parameters['decision_tree'])),
            ('linreg', LinearRegression(**config.model_parameters['linreg']))
        ]

        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(max_iter=1000),
            **params
        )
        model.fit(X_train, y_train)
    
        
    # Model predictions
    pred = model.predict(X_val)
    score = root_mean_squared_error(pred, y_val)


    # Model saving
    if save_model:
        
        # Path to save models
        save_dir = os.path.join(config.path_to_experiment, f'{model_name}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Saving ML models directly; saving custom DNN checkpoints
        checkpoint = {
            'preprocessor': preprocessor,
            'model': model if model_name != 'DNN' else model.get_checkpoint()
        }

        joblib.dump(checkpoint, f"{save_dir}/fold_{fold_index}.joblib")
        

    
    return score

def run_cv(X: pd.DataFrame, y: pd.Series, model_name, config, params=None, save_models=True):
    """Evaluate a model using cross-validation."""
    
    # Cross-Validation strategy
    cv = KFold(**config.kfold_parameters)
    
    val_scores = []
     
    start_time = time.perf_counter()
    for fold_index, (train_indices, val_indices) in enumerate(cv.split(X, y)):
        print(f'Validation {fold_index+1} fold started!')
        
        X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
        X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]
        
        
        set_seed(config.seed)
        
        score = train_one_model(X_train, y_train, X_val, y_val, model_name, fold_index, config, params, save_models)

        val_scores.append(score)
    
    end_time = time.perf_counter()
    cv_time = end_time - start_time
    
    return val_scores, cv_time
      
# Optuna    
def suggest_parameters(trial, search_space):
    """Sample hyperparameters from the search space using Optuna."""
    
    params = {}
    
    methods = {
        'int': trial.suggest_int,
        'float': trial.suggest_float,
        'categorical': trial.suggest_categorical
    }

    for model_name, settings in search_space.items():
        param_type = settings['type']
        kwargs = {k: v for k, v in settings.items() if k != 'type'}

        params[model_name] = methods[param_type](model_name, **kwargs)

    return params

# Optuna
def tune_hyperparameters(X, y, model_name, config):
    """Optimize model hyperparameters using cross-validation."""

    def objective(trial):
        """Evaluate a trial using mean cross-validation RMSE."""
        suggested_params = suggest_parameters(
            trial,
            config.search_spaces[model_name]
        )

        params = {
            **config.model_parameters[model_name],
            **suggested_params
        }

        scores, _ = run_cv(
            X, y, model_name, config,
            params=params,
            save_models=False
        )

        return float(np.mean(scores))

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(
            seed=config.seed
        )
    )

    study.optimize(
        objective,
        n_trials=config.optuna_n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )

    print(f'{model_name} best CV RMSE: {study.best_value:.4f}')
    print(f'{model_name} best parameters: {study.best_params}')

    return study.best_params