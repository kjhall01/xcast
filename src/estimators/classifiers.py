from ..flat_estimators.classifiers import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from .base_estimator import BaseEstimator
from ..preprocessing.onehot import RankedTerciles
from ..preprocessing.spatial import regrid
from ..core.utilities import guess_coords, check_all
import xarray as xr


class cMemberCount:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, an_thresh=0.67, bn_thresh=0.33,  explicit=False):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        X1 = X.sel()  # fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        Y1 = Y.sel()  # fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        self.regrid_coords_lat = Y1.coords[y_lat_dim].values
        self.regrid_coords_lon = Y1.coords[y_lon_dim].values
        if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
            X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim,
                        x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        self.onehots = []
        transformed = []
        for i in range(X1.shape[list(X1.dims).index(x_feature_dim)]):
            dc = {x_feature_dim: i}
            to_transform = X1.isel(**dc)
            to_transform = to_transform.expand_dims(x_feature_dim)
            to_transform.coords[x_feature_dim] = [i]
            self.onehots.append(RankedTerciles(
                low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit))
            self.onehots[i].fit(to_transform, x_lat_dim,
                                x_lon_dim, x_sample_dim, x_feature_dim)

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        X1 = X.sel()  # fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
            X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim,
                        x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

        transformed = []
        for i in range(X1.shape[list(X1.dims).index(x_feature_dim)]):
            dc = {x_feature_dim: i}
            to_transform = X1.isel(**dc)
            to_transform = to_transform.expand_dims(x_feature_dim)
            to_transform.coords[x_feature_dim] = [i]
            transformed_da = self.onehots[i].transform(
                to_transform, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
            transformed.append(transformed_da)

        X1 = xr.concat(transformed, x_feature_dim)
        return X1.sum(x_feature_dim) / X1.sum(x_feature_dim).sum('C')


class cMultivariateLogisticRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = MultivariateELRClassifier


class cExtendedLogisticRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = ELRClassifier


class cMultiLayerPerceptron(BaseEstimator):
    def __init__(self, hidden_layer_sizes=None, **kwargs):
        if hidden_layer_sizes is not None:
            kwargs['hidden_layer_sizes'] = hidden_layer_sizes
        else:
            kwargs['hidden_layer_sizes'] = (5,)
        super().__init__(**kwargs)
        self.model_type = MLPClassifier


class cNaiveBayes(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = NaiveBayesClassifier


class cRandomForest(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = RFClassifier


class cPOELM(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = POELMClassifier
