import enum

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
)
from xgboost import XGBRegressor

from data.encoding import (
    MultiHotEncoder,
    GeoClusteringTransformer,
    DataFrameOutputWrapper,
)


class ModelType(enum.Enum):
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    CATBOOST = "catboost"

    @classmethod
    def make(cls, s: str) -> "ModelType":
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(
                f"Invalid model type: {s}. Choose from {[e.value for e in cls]}"
            )


def make_column_transformer(model_type: ModelType) -> ColumnTransformer:
    num_cols = [
        "square_feet",
        "longitude",
        "latitude",
        "bedrooms",
        "bathrooms",
    ]  # consider transforming bathrooms and bedrooms as categoricals,
    # for random forests also numerical should work
    cat_cols = ["source", "state", "cityname", "has_photo", "fee"]
    # text_col = "text"

    # Preprocessing pipelines
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    if model_type == ModelType.CATBOOST:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ]
        )
    else:
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                ),
            ]
        )

    multi_hot_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("multihot", MultiHotEncoder()),
        ]
    )

    # text_pipeline = make_pipeline(
    #     FunctionTransformer(lambda x: x[text_col], validate=False),
    #     TfidfVectorizer(max_features=1000, stop_words="english"),
    # )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("amen", multi_hot_pipeline, ["amenities"]),
            ("geo", GeoClusteringTransformer(), ["latitude", "longitude", "cityname"]),
            # ("multi_pets", multi_hot_pipeline, ["pets_allowed"]),
            # ("txt", text_pipeline, [text_col]),
            # ("title_tfidf", TfidfVectorizer(max_features=50), "title"),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def make_preprocessing_pipeline(model_type: ModelType) -> Pipeline:

    # id is same as index
    # currency is all USD
    # price_type is all monthly
    COLS_TO_DROP = [
        "id",
        "currency",
        "price_type",
        "body",
        "time",
        "category",  # with exception of < 50 items they are all of one type
    ]

    def drop_columns(X):
        return X.drop(columns=COLS_TO_DROP, errors="ignore")

    def add_text_column(X):
        """Combine title and body into text"""
        X = X.copy()
        X["text"] = X["title"].fillna("") + " " + X["body"].fillna("")
        return X

    preprocessor = make_column_transformer(model_type)
    wrapped_preprocessor = DataFrameOutputWrapper(preprocessor)

    def coerce_numeric_types(X):
        X = X.copy()
        numeric_cols = ["square_feet", "longitude", "latitude", "bedrooms", "bathrooms"]
        amen_cols = [col for col in X.columns if col.startswith("amenity_")]
        all_numeric_cols = numeric_cols.copy()
        all_numeric_cols.extend(amen_cols)
        for col in all_numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    coerce_transformer = FunctionTransformer(coerce_numeric_types, validate=False)

    pipeline = Pipeline(
        steps=[
            # ("add_text", FunctionTransformer(add_text_column, validate=False)),
            ("drop_cols", FunctionTransformer(drop_columns, validate=False)),
            ("preprocessor", wrapped_preprocessor),
            ("coerce_numeric_types", coerce_transformer),
        ]
    )

    return pipeline


def get_model_type_from_model(model: BaseEstimator) -> ModelType:
    if isinstance(model, RandomForestRegressor):
        return ModelType.RANDOM_FOREST
    elif isinstance(model, XGBRegressor):
        return ModelType.XGBOOST
    elif isinstance(model, CatBoostRegressor):
        return ModelType.CATBOOST
    else:
        raise ValueError(f"Unknown model class: {type(model)}")
