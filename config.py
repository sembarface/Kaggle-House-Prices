class Config:
    """Store experiment settings, model parameters, and search spaces."""
    
    seed = 228

    experiment_name = 'default'
    
    models_to_train = [
        "linreg",
        "lasso",
        "ridge",
        "elasticnet",
        "KNN",
        "decision_tree",
        "random_forest",
        "catboost",
        "xgboost",
        "lgbm",
        "DNN",
        "stacking_knn_tree_linreg"
    ]
    
    solver_model = 'catboost'
    get_submission = True
    
    use_holdout = False

    is_train = True
    
    use_optuna = False
    optuna_n_trials = 30
    
    show_plots = False
    show_shap = False
    
    path_to_experiment = f"experiments/{experiment_name}"
    path_to_data = f"data/train.csv"
    path_to_submissions = f'submissions'
    
    
    # feature_parameters = {

    # }
    
    # Preprocessing
    preprocessing = {
        "scale_models": [
            "linreg",
            "ridge",
            "lasso",
            "elasticnet",
            "KNN",
            "DNN",
            "stacking_knn_tree_linreg"
        ],

        "native_categorical_models": [
            "catboost"
        ]
    }
    
    kfold_parameters = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": seed
    }
    
    model_parameters = {
        "linreg": {},

        "ridge": {
            "alpha": 10.0,
            "max_iter": 10000
        },

        "lasso": {
            "alpha": 0.0005,
            "max_iter": 15000
        },

        "elasticnet": {
            "alpha": 0.0005,
            "l1_ratio": 0.5,
            "max_iter": 15000
        },

        "KNN": {
            "n_neighbors": 7,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1
        },

        "decision_tree": {
            "criterion": "squared_error",
            "max_depth": 8,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": seed
        },

        "random_forest": {
            "n_estimators": 400,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": 0.8,
            "n_jobs": -1,
            "random_state": seed
        },

        "catboost": {
            "iterations": 1200,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": seed,
            "one_hot_max_size": 10,
            "early_stopping_rounds": 100,
            "use_best_model": True,
            "verbose": False
        },

        "xgboost": {
            "n_estimators": 1200,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": seed,
            "early_stopping_rounds": 100
        },

        "lgbm": {
            "n_estimators": 1200,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": seed,
            'early_stopping_rounds': 10,
        },

        "DNN": {
            "device": "cpu",
            "output_size": 1,
            "epochs": 200,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "optimizer_name": "Adam"
        },

        "stacking_knn_tree_linreg": {
            "cv": 5,
            "n_jobs": -1,
            "passthrough": False
        }
    }

    search_spaces = {
        "linreg": {},

        "ridge": {
            "alpha": {
                "type": "float",
                "low": 1e-3,
                "high": 100.0,
                "log": True
            }
        },

        "lasso": {
            "alpha": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True
            }
        },

        "elasticnet": {
            "alpha": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True
            },
            "l1_ratio": {
                "type": "float",
                "low": 0.1,
                "high": 0.9
            }
        },

        "KNN": {
            "n_neighbors": {
                "type": "int",
                "low": 3,
                "high": 25
            },
            "weights": {
                "type": "categorical",
                "choices": ["uniform", "distance"]
            },
            "p": {
                "type": "categorical",
                "choices": [1, 2]
            }
        },

        "decision_tree": {
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 15
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 15
            },
            "min_samples_split": {
                "type": "int",
                "low": 2,
                "high": 20
            }
        },

        "random_forest": {
            "max_depth": {
                "type": "int",
                "low": 5,
                "high": 25
            },
            "min_samples_leaf": {
                "type": "int",
                "low": 1,
                "high": 10
            },
            "max_features": {
                "type": "categorical",
                "choices": ["sqrt", 0.5, 0.8, 1.0]
            },
            "n_estimators": {
                "type": "int",
                "low": 200,
                "high": 800,
                "step": 100
            }
        },

        "catboost": {
            "iterations": {
                "type": "int",
                "low": 500,
                "high": 1500,
                "step": 250
            },
            "depth": {
                "type": "int",
                "low": 3,
                "high": 8
            },
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.1,
                "log": True
            },
            "l2_leaf_reg": {
                "type": "float",
                "low": 1.0,
                "high": 10.0,
                "log": True
            },
            "random_strength": {
                "type": "float",
                "low": 0.0,
                "high": 2.0
            }
        },

        "xgboost": {
            "n_estimators": {
                "type": "int",
                "low": 500,
                "high": 1500,
                "step": 250
            },
            "max_depth": {
                "type": "int",
                "low": 2,
                "high": 6
            },
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.1,
                "log": True
            },
            "min_child_weight": {
                "type": "int",
                "low": 1,
                "high": 10
            },
            "reg_lambda": {
                "type": "float",
                "low": 0.1,
                "high": 10.0,
                "log": True
            }
        },

        "lgbm": {
            "n_estimators": {
                "type": "int",
                "low": 500,
                "high": 1500,
                "step": 250
            },
            "num_leaves": {
                "type": "int",
                "low": 5,
                "high": 31
            },
            "min_child_samples": {
                "type": "int",
                "low": 5,
                "high": 40
            },
            "learning_rate": {
                "type": "float",
                "low": 0.01,
                "high": 0.1,
                "log": True
            },
            "reg_lambda": {
                "type": "float",
                "low": 0.0,
                "high": 5.0
            }
        },

        "DNN": {
            "learning_rate": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True
            },
            "batch_size": {
                "type": "categorical",
                "choices": [16, 32, 64]
            },
            "epochs": {
                "type": "int",
                "low": 100,
                "high": 300,
                "step": 100
            },
            "optimizer_name": {
                "type": "categorical",
                "choices": ["Adam", "SGD"]
            }
        },

        "stacking_knn_tree_linreg": {
            "cv": {
                "type": "categorical",
                "choices": [3, 5]
            },
            "passthrough": {
                "type": "categorical",
                "choices": [True, False]
            }
        }
    }

    def to_dict(self):
        """Convert experiment configuration into a dictionary."""

        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "use_optuna": self.use_optuna,
            "optuna_n_trials": self.optuna_n_trials if self.use_optuna else None,
            # "feature_parameters": self.feature_parameters,
            # "preprocessing": self.preprocessing,
            "kfold_parameters": self.kfold_parameters,
            "model_parameters": {
                name: self.model_parameters[name]
                for name in self.models_to_train
            },
            "search_spaces": {
                name: self.search_spaces[name]
                for name in self.models_to_train
            },
            "use_holdout": self.use_holdout,
            "models_to_train": self.models_to_train,
            "solver_model": self.solver_model,
            "get_submission": self.get_submission
        }