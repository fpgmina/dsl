import enum
from functools import partial
from pathlib import Path
from typing import Callable, Any

import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from data.getters import load_X_y, load_X_test
from data.preprocessing import make_preprocessing_pipeline


optuna.logging.set_verbosity(optuna.logging.INFO)


class ModelType(enum.Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"

    @classmethod
    def make(cls, s: str) -> "ModelType":
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(
                f"Invalid model type: {s}. Choose from {[e.value for e in cls]}"
            )


def get_model(model_type: ModelType, **kwargs: Any) -> BaseEstimator:
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
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


def make_pipeline(model: BaseEstimator) -> Pipeline:
    pipeline = make_preprocessing_pipeline()
    pipeline = Pipeline(pipeline.steps + [("model", model)])
    return pipeline


def objective(trial: optuna.trial.Trial, model_type: ModelType) -> float:
    """This function defines the objective (loss) function for Optuna

    Args:
        trial (optuna.trial.Trial): An Optuna trial object
        model_type (ModelType): The model type to use

    Returns:
        float: The loss value to minimize

    Notes:
        Use the trial object to suggest hyperparameter values and then evaluate your model with those values.
        The trial provides suggest_* methods to pick hyperparameter values:
        *  For integers: use trial.suggest_int("param_name", low, high) to sample an integer in [low, high] .
        * For floats: use trial.suggest_float("param_name", low, high, log=True/False) for continuous ranges (set log=True for log-scale sampling) .
        * For categorical choices: use trial.suggest_categorical("param_name", [option1, option2, ...]) to choose from discrete options .
        Each call to a trial.suggest_... defines one hyperparameter of the search space.
         (for example, compute cross-validation score or validation loss).
        The objective function should then return a single value (the metric) for Optuna to minimize or maximize.
    """

    X, y = load_X_y()
    kwargs = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1000),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
    }

    if model_type == ModelType.XGBOOST:
        kwargs["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.1, log=True
        )
        kwargs["reg_lambda"] = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True)
        kwargs["reg_alpha"] = trial.suggest_float("reg_alpha", 0, 1.0)
        kwargs["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        kwargs["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        kwargs["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        kwargs["gamma"] = trial.suggest_float("gamma", 0, 5.0)

    model = get_model(model_type, **kwargs)

    # The trial object effectively decides what value to try next for that parameter. After suggesting values, create a model with suggested hyperparameters
    pipeline = make_pipeline(model)
    # use 3-fold cross-validation
    score = cross_val_score(
        pipeline, X, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()
    return score  # maximize -MAE <==> minimize MAE


def tune(objective: Callable, model_type: ModelType, n_trials=5) -> optuna.study.Study:
    study = optuna.create_study(
        study_name="rfr",
        direction="maximize",
        load_if_exists=True,
    )
    obj = partial(objective, model_type=model_type)
    study.optimize(obj, n_trials=n_trials)
    return study


def fit(study: optuna.study.Study, model_type: ModelType) -> Pipeline:
    X, y = load_X_y()
    try:
        best_params = study.best_params
    except ValueError as e:
        raise RuntimeError("No successful trials found in the Optuna study.") from e

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
