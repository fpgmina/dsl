import argparse

from train import objective, tune, fit, predict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=float, default=2)
    args = parser.parse_args()

    study = tune(objective, n_trials=args.n_trials)
    pipeline = fit(study)
    preds = predict(pipeline, save_to_csv=True)
