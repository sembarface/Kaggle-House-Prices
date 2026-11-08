import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class Preprocessor:

    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config

    def _fit_aggregations(self, X):
        self.area_means_ = X.groupby("Neighborhood")["GrLivArea"].mean()
        self.global_area_ = X["GrLivArea"].mean()

    def _create_features(self, X):
        X = X.copy()

        X["HouseAge"] = X["YrSold"] - X["YearBuilt"]
        X["TotalArea"] = X["1stFlrSF"] + X["2ndFlrSF"]
        
        # Aggregated features
        X["NeighborhoodMeanArea"] = (
            X["Neighborhood"].map(self.area_means_)
            .fillna(self.global_area_)
        )

        X["MSSubClass"] = X["MSSubClass"].astype('category')

        return X

    def _build_transformer(self):
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
                sparse_output=False
            ))
        ])

        return ColumnTransformer([
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(
                dtype_include=["object", "category", "string"]
            ))
        ]).set_output(transform="pandas")

    def fit(self, X, y=None):
        self.input_cols_ = X.columns.tolist()

        self._fit_aggregations(X)
        X = self._create_features(X)

        if self.model_name == "catboost":
            self.cat_cols_ = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
        else:
            self.transformer_ = self._build_transformer()
            self.transformer_.fit(X, y)

        return self

    def transform(self, X):
        X = self._create_features(X[self.input_cols_])

        if self.model_name == "catboost":
            X[self.cat_cols_] = (
                X[self.cat_cols_]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )
            return X

        return self.transformer_.transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)