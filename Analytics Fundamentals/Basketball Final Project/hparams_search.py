import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
import pickle
import os
import time


def generate_params():
    gamma_space = np.logspace(-4, 0, 8).tolist()
    gamma_space.extend(["scale", "auto"])
    params = {
        "Ridge": {
            'Ridge__alpha': np.linspace(0, 2, 21),
            'Ridge__solver': ['auto', 'svd', 'lsqr', 'sag']
        },
        "Lasso": {
            'Lasso__alpha': np.linspace(0, 2, 21),
            'Lasso__max_iter': [100, 1000, 10000]
        },
        "ElasticNet": {
            'ElasticNet__alpha': np.linspace(0, 2, 21),
            "ElasticNet__l1_ratio": np.arange(0, 1, 0.01)
        },
        "KNN": {
            "KNN__n_neighbors": np.arange(1, 31),
            'KNN__weights': ['uniform', 'distance'],
            'KNN__leaf_size': np.arange(1, 50),
            'KNN__p': [1, 2]
        },
        "CART": {
            'CART__max_depth': np.append([1, 5, 10, 15, 20, 25, 30], [None]),
            "CART__min_samples_split": np.arange(1, 31),
            "CART__min_samples_leaf": np.arange(1, 31),
            "CART__max_features": ['auto', 'sqrt', 'log2', None]
        },
        "RandomForestRegressor": {
            'RandomForestRegressor__n_estimators': [100, 500, 1000],
            'RandomForestRegressor__max_depth': np.append([1, 5, 10, 15, 20, 25, 30], [None]),
            "RandomForestRegressor__min_samples_split": np.arange(1, 31),
            "RandomForestRegressor__min_samples_leaf": np.arange(1, 31),
            "RandomForestRegressor__max_features": ['sqrt', 'log2', None, 1]
        },
        "SVR": {
            'SVR__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'SVR__gamma': gamma_space,
            'SVR__C': np.logspace(-0, 4, 8),
            'SVR__epsilon': np.linspace(0.1, 1, 10)
        },
        "XGBoost": {
            'XGBoost__n_estimators': [100, 500, 1000],
            "XGBoost__max_depth": np.append(np.arange(1, 51), [None]),
        },
        "LGBMRegressor": {
            'LGBMRegressor__n_estimators': [100, 500, 1000],
            "LGBMRegressor__max_depth": np.arange(-1, 51),
        }
    }

    return params


def generate_pca_params():
    gamma_space = np.logspace(-4, 0, 8).tolist()
    gamma_space.extend(["scale", "auto"])
    pca_n_components = [None]
    pca_n_components.extend(range(0, 11, 2))
    params = {
        "Ridge": {
            'Ridge__alpha': np.linspace(0, 2, 21),
            'Ridge__solver': ['auto', 'svd', 'lsqr', 'sag'],
            'PCA__n_components': pca_n_components
        },
        "Lasso": {
            'Lasso__alpha': np.linspace(0, 2, 21),
            'Lasso__max_iter': [100, 1000, 10000],
            'PCA__n_components': pca_n_components
        },
        "ElasticNet": {
            'ElasticNet__alpha': np.linspace(0, 2, 21),
            "ElasticNet__l1_ratio": np.arange(0, 1, 0.01),
            'PCA__n_components': pca_n_components
        },
        "KNN": {
            "KNN__n_neighbors": np.arange(1, 31),
            'KNN__weights': ['uniform', 'distance'],
            'KNN__leaf_size': np.arange(1, 50),
            'KNN__p': [1, 2]
        },
        "CART": {
            'CART__max_depth': np.append([1, 5, 10, 15, 20, 25, 30], [None]),
            "CART__min_samples_split": np.arange(1, 31),
            "CART__min_samples_leaf": np.arange(1, 31),
            "CART__max_features": ['auto', 'sqrt', 'log2', None]
        },
        "RandomForestRegressor": {
            'RandomForestRegressor__n_estimators': [100, 500, 1000],
            'RandomForestRegressor__max_depth': np.append([1, 5, 10, 15, 20, 25, 30], [None]),
            "RandomForestRegressor__min_samples_split": np.arange(1, 31),
            "RandomForestRegressor__min_samples_leaf": np.arange(1, 31),
            "RandomForestRegressor__max_features": ['sqrt', 'log2', None, 1],
            'PCA__n_components': pca_n_components
        },
        "SVR": {
            'SVR__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'SVR__gamma': gamma_space,
            'SVR__C': np.logspace(-0, 4, 8),
            'SVR__epsilon': np.linspace(0.1, 1, 10)
        },
        "XGBoost": {
            'XGBoost__n_estimators': [100, 500, 1000],
            "XGBoost__max_depth": np.append(np.arange(1, 51), [None]),
            'PCA__n_components': pca_n_components
        },
        "LGBMRegressor": {
            'LGBMRegressor__n_estimators': [100, 500, 1000],
            "LGBMRegressor__max_depth": np.arange(-1, 51),
            'PCA__n_components': pca_n_components
        }
    }

    return params


def hyperparameter_optimization(X, y, pipelines, models_params, cv=5, gs_cv=3, scoring="neg_mean_squared_error", save_path=None):
    best_models = {}
    for model in pipelines:

        if(save_path is not None):
            model_data = hs_params_load_model(
                X, y, cv, scoring, save_path, model)
            if(model_data is not None):
                best_models[model] = model_data
                continue

        print(f'{model} Hyperparameter Tuning...')
        if model == "LinearRegression":
            print(f"Not Hyperparameter Tuning for {model}")
            regressor = Pipeline(pipelines[model]["steps"])
            rmse = np.mean(
                np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring=scoring)))
            best_models[model] = {
                "model": regressor,
                "rmse": rmse,
                "params": None
            }
            continue
        start = time.time()

        regressor = Pipeline(pipelines[model]["steps"])

        if(model == "RandomForestRegressor"):
            gs_best = RandomizedSearchCV(
                regressor, models_params[model], cv=gs_cv, n_jobs=-1, verbose=True, random_state=42).fit(X, y)
        elif(model == "XGBoost" or model == "LGBMRegressor" or model == "SVR"):
            gs_best = HalvingGridSearchCV(
                regressor, models_params[model], cv=gs_cv, n_jobs=-1, verbose=True).fit(X, y)
        else:
            gs_best = GridSearchCV(
                regressor, models_params[model], cv=gs_cv, n_jobs=-1, verbose=True).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        save_model(final_model, save_path + model + ".pickle")

        rmse = np.mean(
            np.sqrt(-cross_val_score(final_model, X, y, cv=cv, scoring=scoring)))
        print(f'RMSE: {rmse}')
        print(f'{model} best params: {gs_best.best_params_}')

        best_models[model] = {
            "model": final_model,
            "rmse": rmse,
            "params": gs_best.best_params_
        }
        end = time.time()
        print(f'{model} took {end - start} seconds')

    return best_models


def hs_params_load_model(X, y, cv, scoring, save_path, model):
    is_saved_model = os.path.exists(save_path + model + ".pickle")
    if(is_saved_model):
        with open(save_path + model + ".pickle", "rb") as f:
            print('Loading ' + model + ' from pickle file...')
            loaded_model = pickle.load(f)
            rmse = np.mean(
                np.sqrt(-cross_val_score(loaded_model, X, y, cv=cv, scoring=scoring)))
            best_params = loaded_model.get_params()

            print(f'RMSE: {rmse}')
            print(f'{model} best params: {best_params}')

            return {
                "model": loaded_model,
                "rmse": rmse,
                "params": best_params
            }
    return None


def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))
