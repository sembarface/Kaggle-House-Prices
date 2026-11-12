import os
import json
import subprocess

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from Solver import Solver


from train import  run_cv, tune_hyperparameters

from config import Config



def main():
    """Run the ML pipeline for training, evaluation, and inference."""
    
    config = Config()
    
    data = pd.read_csv(config.path_to_data, index_col='Id')
    
    # Log dataset target
    X, y = data.drop('SalePrice', axis=1), np.log(data['SalePrice'])
    
    if config.use_holdout:
        # Test samples for final evaluating
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=config.seed)
    

    # Train all models with Cross-Validation
    if config.is_train:
        
        # Tune hyperparameters
        if config.use_optuna:
            for model_name in config.models_to_train:
                best_params = tune_hyperparameters(X, y, model_name, config)
                config.model_parameters = {
                    **config.model_parameters,
                    model_name: {
                        **config.model_parameters[model_name],
                        **best_params
                    }
                }
            
        metrics = {}
        for model_name in config.models_to_train:
            
            val_scores, cv_time = run_cv(X, y, model_name, config)
            
            mean_cv_score = np.mean(val_scores)
            print(f'{model_name} CV mean score = {mean_cv_score:.4f}')
            print(f'{model_name} CV scores std = {np.std(val_scores):.4f}')
            print(f'{model_name} CV time = {cv_time:.4f}\n')
            metrics[model_name] = mean_cv_score
            
            # Logging in results.txt
            with open('results.txt', 'a') as f:
                f.write(f'{config.experiment_name} | {model_name} | CV score {mean_cv_score:.4f} | {cv_time:.4f} s\n')
        
        best_model = min(metrics, key=metrics.get)
        print(f'\nBest model: {best_model}, CV score: {metrics[best_model]:.4f}')
        
        # Сertain experiment config
        experiment_config = config.to_dict()
        # Trying to save commit hash
        try:
            experiment_config['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                text=True
            ).strip()
        except(subprocess.CalledProcessError, FileNotFoundError):
            print('No Git')
            experiment_config['git_commit'] = None
            
            
        # Saving experiment config
        with open(f'{config.path_to_experiment}/config.json', 'w') as f:
            json.dump(experiment_config, f, indent=4)
        
        # Saving metrics
        with open(f'{config.path_to_experiment}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)                  
             

    solver = Solver(config.path_to_experiment, config.solver_model)
    # Train/val plots
    if config.show_plots:
        solver.get_plots()
        
    # Test score
    if config.use_holdout:
        # Pick solver
        solver = Solver(config.path_to_experiment, config.solver_model)
        pred = solver.predict(X_test)
        print(f'{solver.model_name} rmse score = ', mean_squared_error(pred, y_test))
        
        # Train/val plots
        if config.show_plots:
            solver.get_plots()
            
        # Shap beeswarm on train samples
        if config.show_shap:
            X = data.drop('SalePrice', axis=1)
            solver.get_shap(X)
    
    
    # Kaggle submission
    if config.get_submission:
        kaggle_test = pd.read_csv('data/test.csv', index_col='Id')
        # Pick solver
        solver = Solver(config.path_to_experiment, config.solver_model)
        pred = np.exp(solver.predict(kaggle_test))
        submission = pd.DataFrame({
            'Id': kaggle_test.index,
            'SalePrice': pred
        })
        os.makedirs(config.path_to_submissions, exist_ok=True)
        submission.to_csv(f'{config.path_to_submissions}/{config.experiment_name}_{solver.model_name}.csv', index=False)
        

if __name__ == '__main__':
    main()