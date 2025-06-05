import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)


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
    n_estimators = trial.suggest_int(
        "n_estimators", 50, 200
    )  # integer from 50 to 200
    max_depth = trial.suggest_int("max_depth", 2, 10)  # integer from 2 to 10
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])

    # The trial object effectively decides what value to try next for that parameter. After suggesting values, create a model with suggested hyperparameters
    model = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=42, n_jobs=-1
    )
    # and evaluate it (use 3-fold cross-validation and return the mean accuracy)
    score = cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()
    return score  # Optuna will try to maximize this score

