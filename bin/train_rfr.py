import argparse

from train import objective, tune, fit, predict
from data.preprocessing import ModelType

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=float, default=2)
    parser.add_argument(
        "--model_type",
        type=ModelType.make,  # Use the enum's custom method as the converter
        required=True,
        help="Model type to use",
    )
    parser.add_argument("--save_to_csv", type=bool, default=True)
    args = parser.parse_args()
    model_type = args.model_type
    study = tune(objective, n_trials=args.n_trials, model_type=model_type)
    pipeline = fit(study, model_type=model_type)
    preds = predict(pipeline, save_to_csv=args.save_to_csv)


# python -m bin.train_rfr --n_trials 1 --model_type xgboost
