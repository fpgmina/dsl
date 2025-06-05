import optuna

from train import objective


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    #  run the optimization for 20 trials
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)
