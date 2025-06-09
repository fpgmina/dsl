from typing import Tuple

import pandas as pd
import pathlib

from data.preprocessing import make_preprocessing_pipeline, ModelType

ROOT_PATH = pathlib.Path("/Volumes/Samsung SSD 990 PRO 1TB/data/dsl")


def get_train_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT_PATH / "development.csv")
    return df


def get_test_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT_PATH / "evaluation.csv")
    return df


def load_X_y() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = get_train_data()
    X = df.copy()
    y = X.pop("price")
    return X, y


def load_X_test() -> pd.DataFrame:
    df = get_test_data()
    return df


def get_transformed_X(model_type: ModelType) -> pd.DataFrame:
    X, y = load_X_y()
    preproc_pipeline = make_preprocessing_pipeline(model_type)

    X_transformed_df = preproc_pipeline.fit_transform(X)

    return X_transformed_df
