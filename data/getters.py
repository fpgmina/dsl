import pandas as pd
import pathlib


ROOT_PATH = pathlib.Path("/Volumes/Samsung SSD 990 PRO 1TB/data/dsl")




def get_train_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT_PATH / "development.csv")
    return df


def get_test_data() -> pd.DataFrame:
    df = pd.read_csv(ROOT_PATH / "evaluation.csv")
    return df
