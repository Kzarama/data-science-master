import pandas as pd
import numpy as np

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# Pipelines

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA


def train_models(X, y, pipelines, cv=5):
    df = pd.DataFrame(columns=["model", "rmse"])

    for model in pipelines:
        regressor = Pipeline(pipelines[model]["steps"])

        rmse = np.mean(
            np.sqrt(-cross_val_score(regressor, X, y, cv=cv,
                    scoring="neg_mean_squared_error"))
        )

        df = df.append(
            {
                "model": model,
                "regressor": regressor,
                "rmse": rmse
            },
            ignore_index=True
        )

    return df


def base_models_pipelines():
    base_models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "steps": [
                ("LinearRegression", LinearRegression())
            ]
        },
        "Ridge": {
            "model": Ridge(),
            "steps": [
                ("Ridge", Ridge())
            ]
        },
        "Lasso": {
            "model": Lasso(),
            "steps": [
                ("Lasso", Lasso())
            ]
        },
        "ElasticNet": {
            "model": ElasticNet(),
            "steps": [
                ("ElasticNet", ElasticNet())
            ]
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "steps": [
                ("KNN", KNeighborsRegressor())
            ]
        },
        "CART": {
            "model": DecisionTreeRegressor(),
            "steps": [
                ("CART", DecisionTreeRegressor())
            ]
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(),
            "steps": [
                ("RandomForestRegressor", RandomForestRegressor())
            ]
        },
        "SVR": {
            "model": SVR(),
            "steps": [
                ("SVR", SVR())
            ]
        },
        "XGBoost": {
            "model": XGBRegressor(objective='reg:squarederror'),
            "steps": [
                ("XGBoost", XGBRegressor(objective='reg:squarederror'))
            ]
        },
        "LGBMRegressor": {
            "model": LGBMRegressor(),
            "steps": [
                ("LGBMRegressor", LGBMRegressor())
            ]
        }
    }

    return base_models


def models_pca_pipelines():
    pca_models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "steps": [
                ("PCA", PCA()),
                ("LinearRegression", LinearRegression())
            ]
        },
        "Ridge": {
            "model": Ridge(),
            "steps": [
                ("PCA", PCA()),
                ("Ridge", Ridge())
            ]
        },
        "Lasso": {
            "model": Lasso(),
            "steps": [
                ("PCA", PCA()),
                ("Lasso", Lasso())
            ]
        },
        "ElasticNet": {
            "model": ElasticNet(),
            "steps": [
                ("PCA", PCA()),
                ("ElasticNet", ElasticNet())
            ]
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "steps": [
                ("PCA", PCA()),
                ("KNN", KNeighborsRegressor())
            ]
        },
        "CART": {
            "model": DecisionTreeRegressor(),
            "steps": [
                ("PCA", PCA()),
                ("CART", DecisionTreeRegressor())
            ]
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(),
            "steps": [
                ("PCA", PCA()),
                ("RandomForestRegressor", RandomForestRegressor())
            ]
        },
        "SVR": {
            "model": SVR(),
            "steps": [
                ("PCA", PCA()),
                ("SVR", SVR())
            ]
        },
        "XGBoost": {
            "model": XGBRegressor(objective='reg:squarederror'),
            "steps": [
                ("PCA", PCA()),
                ("XGBoost", XGBRegressor(objective='reg:squarederror'))
            ]
        },
        "LGBMRegressor": {
            "model": LGBMRegressor(),
            "steps": [
                ("PCA", PCA()),
                ("LGBMRegressor", LGBMRegressor())
            ]
        }
    }

    return pca_models


def models_data_scaling_pipelines():
    models = {
        "LinearRegression": {
            "model": LinearRegression(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("LinearRegression", LinearRegression())
            ]
        },
        "Ridge": {
            "model": Ridge(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("Ridge", Ridge())
            ]
        },
        "Lasso": {
            "model": Lasso(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("Lasso", Lasso())
            ]
        },
        "ElasticNet": {
            "model": ElasticNet(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("ElasticNet", ElasticNet())
            ]
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("KNN", KNeighborsRegressor())
            ]
        },
        "CART": {
            "model": DecisionTreeRegressor(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("CART", DecisionTreeRegressor())
            ]
        },
        "RandomForestRegressor": {
            "model": RandomForestRegressor(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("RandomForestRegressor", RandomForestRegressor())
            ]
        },
        "SVR": {
            "model": SVR(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("SVR", SVR())
            ]
        },
        "XGBoost": {
            "model": XGBRegressor(objective='reg:squarederror'),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("XGBoost", XGBRegressor(objective='reg:squarederror'))
            ]
        },
        "LGBMRegressor": {
            "model": LGBMRegressor(),
            "steps": [
                ("StandardScaler", StandardScaler()),
                ("LGBMRegressor", LGBMRegressor())
            ]
        }
    }

    return models
