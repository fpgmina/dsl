# DSL Regression Pipeline

This repository provides a command-line pipeline for training and evaluating regression models 
using Optuna for hyperparameter tuning.

## How to Run

Activate your environment and run:

```bash
python -m bin.fit_predict --n_trials 20 --model_type xgboost --save_to_csv True
```

### Arguments

- `--n_trials`: Number of Optuna optimization trials (default: 2)
- `--model_type`: Model type to use (e.g., `xgboost`)
- `--save_to_csv`: Whether to save predictions to CSV (default: True)

Outputs include prediction CSVs and diagnostic plots (e.g., feature importance, optimization history).
