import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.utils.validation import check_is_fitted


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def _preprocess(self, X):
        X = pd.Series(np.ravel(X))  # ensure it's a flat Series
        return X.fillna("").apply(
            lambda s: [i.strip() for i in s.split(",") if i.strip()]
        )

    def fit(self, X, y=None):
        X_list = self._preprocess(X)
        self.mlb.fit(X_list)
        self.classes_ = self.mlb.classes_
        return self

    def transform(self, X):
        check_is_fitted(self, "classes_")
        X_list = self._preprocess(X)
        return self.mlb.transform(X_list)

    def get_feature_names_out(self, input_features=None):
        return [f"amenity_{a}" for a in self.mlb.classes_]


class GeoClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col="latitude", lon_col="longitude", city_col="cityname"):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.city_col = city_col
        self.cluster_models_ = {}
        self.cluster_offsets_ = {}

    def fit(self, X, y=None):
        self.cluster_models_ = {}
        self.cluster_offsets_ = {}
        offset = 0

        # for each city fit clusters
        for city, group in X.groupby(self.city_col):
            coords = group[[self.lat_col, self.lon_col]].dropna()
            # coordinates could be duplicated e.g if there are two apartments in the same building
            unique_coords = coords.drop_duplicates()
            n = len(coords)

            # skip city with less than 30 listings
            if n < 30:
                self.cluster_models_[city] = None
                self.cluster_offsets_[city] = -1
                continue

            if len(unique_coords) < 2:
                self.cluster_models_[city] = None
                self.cluster_offsets_[city] = -1
                continue

            k = max(2, min(int(np.sqrt(n)), len(unique_coords)))
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(coords)

            self.cluster_models_[city] = model  # save the model for the city
            self.cluster_offsets_[city] = offset  # save the offset for the city
            offset += k  # increment by the number of clusters of the city

        return self

    def transform(self, X):
        check_is_fitted(self, "cluster_models_")
        cluster_ids = pd.Series(-1, index=X.index)

        for city, group in X.groupby(self.city_col):
            model = self.cluster_models_.get(city)
            offset = self.cluster_offsets_.get(city, -1)
            coords = group[[self.lat_col, self.lon_col]].dropna()

            if model is not None and not coords.empty:
                preds = model.predict(coords) + offset
                cluster_ids.loc[coords.index] = preds

        return pd.DataFrame({"geo_cluster": cluster_ids}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return ["geo_cluster"]


class DataFrameOutputWrapper(TransformerMixin, BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        Xt = self.transformer.transform(X)
        feature_names = self.transformer.get_feature_names_out()
        return pd.DataFrame(Xt, columns=feature_names, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self.transformer.get_feature_names_out(input_features)


class KeywordFlagger(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keywords = {
            "luxury": r"\bluxur(?:y|ious|ies)\b",
            "penthouse": r"\bpenthouse\b",
            "exclusive": r"\bexclusive\b",
            "high_end": r"\bhigh[-_\s]?end\b",
            "renovated": r"\brenovated\b",
            "updated": r"\bupdated\b",
            "spacious": r"\bspacious\b",
            "open_plan": r"\bopen[-_\s]?plan\b",
            "sunlight": r"\bsunlight|sun-filled|sunny\b",
            "natural_light": r"\bnatural[-_\s]?light\b",
            "modern": r"\bmodern\b",
            "designer": r"\bdesigner\b",
            "pool": r"\bpool\b",
            "doorman": r"\bdoorman\b",
        }
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.feature_names_ = [
            f"{col}_contains_{kw}" for kw in self.keywords for col in ["title", "body"]
        ]
        return self

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        feature_names = []
        for name, pattern in self.keywords.items():
            for col in ["title", "body"]:
                fname = f"{col}_contains_{name}"
                X_new[fname] = (
                    X[col].str.contains(pattern, case=False, na=False).astype(int)
                )
                feature_names.append(fname)
        self.feature_names_ = feature_names  # Ensure it's always set
        return X_new

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "feature_names_"):
            raise AttributeError(
                "Transformer has not been fitted yet. Call fit or fit_transform first."
            )
        return self.feature_names_
