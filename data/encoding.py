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
