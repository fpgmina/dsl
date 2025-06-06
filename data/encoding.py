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
        return self

    def transform(self, X):
        X_list = self._preprocess(X)
        return self.mlb.transform(X_list)

    def get_feature_names_out(self, input_features=None):
        return [f"amenity_{a}" for a in self.mlb.classes_]


class GeoClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col="latitude", lon_col="longitude", city_col="cityname"):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.city_col = city_col
        self.cluster_models_ = {}  # city -> KMeans or None (for default)

    def fit(self, X, y=None):
        X = X.copy()
        self.cluster_models_ = {}

        for city, group in X.groupby(self.city_col):
            coords = group[[self.lat_col, self.lon_col]].dropna()
            n = len(coords)

            if n < 100:
                self.cluster_models_[city] = None  # mark as default cluster
                continue

            k = max(2, min(10, int(np.sqrt(n / 2))))
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(coords)
            self.cluster_models_[city] = kmeans

        return self

    def transform(self, X):
        check_is_fitted(self, "cluster_models_")
        X = X.copy()
        X["geo_cluster"] = np.nan

        for city, group in X.groupby(self.city_col):
            coords = group[[self.lat_col, self.lon_col]]
            indices = group.index

            model = self.cluster_models_.get(city)
            if model is None:
                X.loc[indices, "geo_cluster"] = f"{city}_default_cluster"
                continue

            valid_coords = coords.dropna()
            if valid_coords.empty:
                X.loc[indices, "geo_cluster"] = f"{city}_default_cluster"
                continue

            preds = model.predict(valid_coords)
            labels = [f"{city}_cluster{cid}" for cid in preds]
            X.loc[valid_coords.index, "geo_cluster"] = labels

        return X[["geo_cluster"]]

    def get_feature_names_out(self, input_features=None):
        return ["geo_cluster"]
