from typing import Callable, Tuple

import optuna
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from data.getters import load_X_y, load_X_test
from data.preprocessing import make_preprocessing_pipeline


optuna.logging.set_verbosity(optuna.logging.INFO)


def get_model(n_estimators: int, max_depth: int) -> BaseEstimator:

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        criterion="squared_error",
        random_state=42,
        n_jobs=-1,
    )
    return model


def make_pipeline(model: BaseEstimator) -> Pipeline:
    pipeline = make_preprocessing_pipeline()
    pipeline = Pipeline(pipeline.steps + [("model", model)])
    return pipeline


# 1. Define the objective function
def objective(trial: optuna.trial.Trial) -> float:
    """This function defines the objective (loss) function for Optuna

    Args:
        trial (optuna.trial.Trial): An Optuna trial object

    Returns:
        float: The loss value to minimize

    Comments:
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
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 4, 20)  # integer from 2 to 10

    print(
        f"Trial {trial.number} | Params: "
        f"n_estimators={n_estimators}, "
        f"max_depth={max_depth}"
    )

    # The trial object effectively decides what value to try next for that parameter. After suggesting values, create a model with suggested hyperparameters
    model = get_model(n_estimators=n_estimators, max_depth=max_depth)
    pipeline = make_pipeline(model)
    # use 3-fold cross-validation
    score = cross_val_score(
        pipeline, X, y, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()
    return score  # maximize -MAE <==> minimize MAE


def tune(objective: Callable, n_trials=5) -> optuna.study.Study:
    study = optuna.create_study(
        study_name="rfr",
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def fit(study: optuna.study.Study) -> Pipeline:
    X, y = load_X_y()
    try:
        best_params = study.best_params
    except ValueError as e:
        raise RuntimeError("No successful trials found in the Optuna study.") from e

    model = get_model(
        n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"]
    )
    pipeline = make_pipeline(model)

    pipeline.fit(X, y)
    return pipeline


def predict(pipeline: Pipeline, save_to_csv: bool = False) -> None | pd.DataFrame:
    X_test = load_X_test()
    preds = pipeline.predict(X_test)
    df_preds = pd.DataFrame({"Id": range(len(preds)), "Predicted": preds})
    if save_to_csv:
        df_preds.to_csv("../preds/predictions.csv", index=False)
    else:
        return preds


def compute_r2(study: optuna.study.Study) -> Tuple[float, float]:
    X, y = load_X_y()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    try:
        best_params = study.best_params
    except ValueError as e:
        raise RuntimeError("No successful trials found in the Optuna study.") from e

    model = get_model(
        n_estimators=best_params["n_estimators"], max_depth=best_params["max_depth"]
    )
    pipeline = make_pipeline(model)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return r2, mae
