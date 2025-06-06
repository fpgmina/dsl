from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
)

from data.encoding import MultiHotEncoder, GeoClusteringTransformer


def make_column_transformer() -> ColumnTransformer:
    num_cols = [
        "square_feet",
        "longitude",
        "latitude",
        "bedrooms",
        "bathrooms",
    ]  # consider transforming bathrooms and bedrooms as categoricals,
    # for random forests also numerical should work
    cat_cols = [
        "category",
        "source",
        "state",
        "cityname",
        "has_photo",
    ]
    # text_col = "text"

    # Preprocessing pipelines
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    #
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
        ]
    )

    return preprocessor


def make_preprocessing_pipeline() -> Pipeline:

    # id is same as index
    # currency is all USD
    # price_type is all monthly
    COLS_TO_DROP = [
        "id",
        "currency",
        "price_type",
        "body",
        "time",
        "title",
    ]

    def drop_columns(X):
        return X.drop(columns=COLS_TO_DROP, errors="ignore")

    def add_text_column(X):
        """Combine title and body into text"""
        X = X.copy()
        X["text"] = X["title"].fillna("") + " " + X["body"].fillna("")
        return X

    pipeline = Pipeline(
        steps=[
            # ("add_text", FunctionTransformer(add_text_column, validate=False)),
            ("drop_cols", FunctionTransformer(drop_columns, validate=False)),
            ("preprocessor", make_column_transformer()),
        ]
    )

    return pipeline
