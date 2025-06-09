from functools import partial
from pathlib import Path
from typing import Callable, Any

import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from data.getters import load_X_y, load_X_test
from data.preprocessing import (
    make_preprocessing_pipeline,
    ModelType,
    get_model_type_from_model,
)

optuna.logging.set_verbosity(optuna.logging.INFO)


RegressorType = RandomForestRegressor | CatBoostRegressor | XGBRegressor


def get_model(model_type: ModelType, **kwargs: Any) -> RegressorType:
    if model_type == ModelType.RANDOM_FOREST:
        model = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators"),
            max_depth=kwargs.get("max_depth"),
            criterion=kwargs.get("criterion", "squared_error"),
            random_state=kwargs.get("random_state", 42),
            n_jobs=kwargs.get("n_jobs", -1),
        )

    elif model_type == ModelType.XGBOOST:
        model = XGBRegressor(
            n_estimators=kwargs.get("n_estimators"),
            max_depth=kwargs.get("max_depth"),
            learning_rate=kwargs.get("learning_rate"),
            objective=kwargs.get("objective", "reg:squarederror"),
            tree_method=kwargs.get("tree_method", "hist"),
            n_jobs=kwargs.get("n_jobs", -1),
            random_state=kwargs.get("random_state", 42),
            early_stopping_rounds=kwargs.get("early_stopping_rounds", 50),
        )
    elif model_type == ModelType.CATBOOST:
        return CatBoostRegressor(
            iterations=kwargs.get("n_estimators"),
            depth=kwargs.get("max_depth"),
            learning_rate=kwargs.get("learning_rate", 0.1),
            l2_leaf_reg=kwargs.get("reg_lambda", 3.0),
            verbose=0,
            cat_features=kwargs.get("cat_features"),  # this is key
            random_state=kwargs.get("random_state", 42),
            thread_count=kwargs.get("thread_count", 12),
            early_stopping_rounds=kwargs.get("early_stopping_rounds", 50),
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def make_pipeline(model: BaseEstimator) -> Pipeline:
    model_type = get_model_type_from_model(model)
    pipeline = make_preprocessing_pipeline(model_type)
    pipeline = Pipeline(pipeline.steps + [("model", model)])
    return pipeline


def objective(trial: optuna.trial.Trial, model_type: ModelType) -> float:
    """
    This function defines the objective (loss) function for Optuna
        The objective function should then return a single value (the metric) for Optuna
        to minimize or maximize.
    """

    X, y = load_X_y()
    kwargs = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1000),
    }

    if model_type == ModelType.XGBOOST:
        kwargs["max_depth"] = trial.suggest_int("max_depth", 4, 20)
        kwargs["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        )
        kwargs["reg_lambda"] = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)
        kwargs["reg_alpha"] = trial.suggest_float("reg_alpha", 0, 1.0)
        kwargs["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        kwargs["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        kwargs["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        # kwargs["gamma"] = trial.suggest_float("gamma", 0, 5.0)

    if model_type == ModelType.CATBOOST:
        kwargs["max_depth"] = trial.suggest_int(
            "max_depth", 4, 16
        )  # maximum tree depth is 16 for catboost
        kwargs["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        )
        kwargs["reg_lambda"] = trial.suggest_float("reg_lambda", 1, 10.0, log=True)
        kwargs["cat_features"] = [
            "state",
            "cityname",
            "has_photo",
            "fee",
            "geo_cluster",
        ]

    model = get_model(model_type, **kwargs)

    pipeline = make_pipeline(model)
    # use 3-fold cross-validation
    score = cross_val_score(
        pipeline, X, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()
    return score  # maximize -MAE <==> minimize MAE


def tune(
    objective_func: Callable, model_type: ModelType, n_trials=5
) -> optuna.study.Study:
    study = optuna.create_study(
        study_name=f"regression_{model_type.value}",
        direction="maximize",
        load_if_exists=True,
    )
    obj = partial(objective_func, model_type=model_type)
    study.optimize(obj, n_trials=n_trials)
    return study


def fit(study: optuna.study.Study, model_type: ModelType) -> Pipeline:
    X, y = load_X_y()
    try:
        best_params = study.best_params
    except ValueError as e:
        raise RuntimeError("No successful trials found in the Optuna study.") from e
    print("Best hyperparameters:")
    print(study.best_params)
    model = get_model(model_type=model_type, **best_params)
    pipeline = make_pipeline(model)
    pipeline.fit(X, y)
    return pipeline


def predict(pipeline: Pipeline, save_to_csv: bool = False) -> None | pd.DataFrame:
    X_test = load_X_test()
    try:
        check_is_fitted(pipeline)
    except NotFittedError:
        raise RuntimeError("Pipeline was not fitted before calling predict().")
    preds = pipeline.predict(X_test)
    df_preds = pd.DataFrame({"Id": range(len(preds)), "Predicted": preds})
    if save_to_csv:
        output_dir = Path.cwd() / "preds"  # Use current working directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predictions.csv"
        df_preds.to_csv(output_path, index=False)
        return None
    else:
        return preds
