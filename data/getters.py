import pandas as pd


def get_train_data() -> pd.DataFrame:
    df = pd.read_csv("/Volumes/Samsung SSD 990 PRO 1TB/data/dsl/development.csv")
    return df


def get_test_data() -> pd.DataFrame:
    df = pd.read_csv("/Volumes/Samsung SSD 990 PRO 1TB/data/dsl/evaluation.csv")
    return df
