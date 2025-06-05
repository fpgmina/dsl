import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler


def make_column_transformer() -> ColumnTransformer:
    num_cols = ["square_feet", "bathrooms", "bedrooms"]
    cat_cols = ["category", "currency", "price_type", "source", "state", "cityname"]
    bool_cols = ["has_photo", "pets_allowed"]
    text_col = "text"

    # Preprocessing pipelines
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    bool_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("to_float", FunctionTransformer(lambda x: x.astype(float))),
        ]
    )

    text_pipeline = make_pipeline(
        FunctionTransformer(lambda x: x[text_col], validate=False),
        TfidfVectorizer(max_features=1000, stop_words="english"),
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("bool", bool_pipeline, bool_cols),
            ("txt", text_pipeline, [text_col]),
        ]
    )

    return preprocessor


def make_preprocessing_pipeline() -> Pipeline:

    COLS_TO_DROP = ["id", "currency", "price_type", "title", "body", "time"]

    def drop_columns(X):
        return X.drop(columns=COLS_TO_DROP, errors="ignore")

    def add_text_column(X):
        """Combine title and body into text"""
        X = X.copy()
        X["text"] = X["title"].fillna("") + " " + X["body"].fillna("")
        return X

    pipeline = Pipeline(
        steps=[
            ("add_text", FunctionTransformer(add_text_column, validate=False)),
            ("drop_cols", FunctionTransformer(drop_columns, validate=False)),
            ("preprocessor", make_column_transformer()),
        ]
    )

    return pipeline
