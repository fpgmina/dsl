import optuna
import xgboost
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
import optuna.visualization.matplotlib as optuna_viz


def xgboost_feature_importance(
    pipeline: Pipeline, filename: str = "feature_importance.png"
) -> None:
    """
    Extract the XGBoost model from a scikit-learn pipeline and save the feature importance plot.
    """
    model = pipeline.steps[-1][1]

    if not isinstance(model, xgboost.XGBModel):
        raise ValueError("The last step in the pipeline is not an XGBoost model.")

    fig, ax = plt.subplots(figsize=(10, 8))
    xgboost.plot_importance(
        model,
        ax=ax,
        importance_type="gain",
        title="XGBoost Feature Importance",
        xlabel="Gain",
        show_values=False,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_optimization_history(
    study: optuna.study.Study, filename: str = "optuna_opt_history"
) -> None:
    fig = optuna_viz.plot_optimization_history(study)
    fig.set_size_inches(10, 6)
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
