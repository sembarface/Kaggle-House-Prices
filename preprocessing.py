import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class Preprocessor:
    """Preprocess tabular features for training and inference."""
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config

    def _fit_aggregations(self, X: pd.DataFrame):
        """Calculate feature statistics on training data."""
        
        params = self.config.feature_parameters
        
        # Calculate neighborhood statistics using training data only
        if params['add_NeighborhoodMeanArea']:
            self.area_means_ = X.groupby("Neighborhood")["GrLivArea"].mean()
            self.global_area_ = X["GrLivArea"].mean()

    def _create_features(self, X: pd.DataFrame):
        """Create new features from existing columns."""
        
        X = X.copy()
        params = self.config.feature_parameters
        
        # Fixing wrong column type
        X["MSSubClass"] = X["MSSubClass"].astype('category')
        
        # New features
        if params['add_HouseAge']:
            X["HouseAge"] = X["YrSold"] - X["YearBuilt"]

        if params['add_TotalArea']:   
            X["TotalArea"] = X["1stFlrSF"] + X["2ndFlrSF"]
        
        # Aggregated features
        # Use global mean for unseen neighborhoods
        if params['add_NeighborhoodMeanArea']:
            X["NeighborhoodMeanArea"] = (
                X["Neighborhood"].map(self.area_means_)
                .fillna(self.global_area_)
            )
        
        X = X.drop(columns=params['drop_columns'])
        return X
    
    def _fill_missing(self, X: pd.DataFrame):
        """Fill missing values with feature-specific values."""
        
        # Missing values represent the absence of an object
        X = X.copy()
        none_categories = self.config.feature_parameters['none_categories']
        zero_fill_columns = self.config.feature_parameters['zero_fill_columns']
        X[none_categories] = X[none_categories].fillna('None')
        X[zero_fill_columns] = X[zero_fill_columns].fillna(0)

        return X
        

    def _build_transformer(self):
        """Build numerical and categorical preprocessing pipelines."""
        
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

        if self.model_name in self.config.preprocessing["scale_models"]:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric = Pipeline(numeric_steps)

        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10,
                sparse_output=False
            ))
        ])

        return ColumnTransformer([
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(
                dtype_include=["object", "category", "string"]
            ))
        ]).set_output(transform="pandas")

    def fit(self, X: pd.DataFrame, y=None):
        """Fit preprocessing statistics and transformations."""
        
        self.input_cols_ = X.columns.tolist()

        X = self._fill_missing(X)
        self._fit_aggregations(X)
        X = self._create_features(X)

        if self.model_name == 'catboost':
            self.cat_cols_ = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        else:
            self.transformer_ = self._build_transformer()
            self.transformer_.fit(X, y)

        return self

    def transform(self, X: pd.DataFrame):
        """Transform data using the fitted preprocessor."""
                
        X = self._fill_missing(X[self.input_cols_])
        X = self._create_features(X[self.input_cols_])

        
        if self.model_name == 'catboost':
            X[self.cat_cols_] = (
                X[self.cat_cols_]
                .astype('string')
                .fillna("__MISSING__")
                .astype(str)
            )
            return X

        return self.transformer_.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None):
        """Fit the preprocessor and transform training data."""
        return self.fit(X, y).transform(X)