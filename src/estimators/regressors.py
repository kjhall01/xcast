import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from ..flat_estimators.regressors import PoissonRegressionOne, GammaRegressionOne, ELMRegressor, RandomForestRegressorOne
from ..flat_estimators import EinsteinLearningMachine
from .base_estimator import BaseEstimator
from ..preprocessing.spatial import regrid
from ..preprocessing.normal import Normal
from ..core.utilities import check_all, guess_coords


class EnsembleMean:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False,  parallel_in_memory=True):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

        X1 = X.sel()  # fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        Y1 = Y.sel()  # fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        self.regrid_coords_lat = Y1.coords[y_lat_dim].values
        self.regrid_coords_lon = Y1.coords[y_lon_dim].values
        if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
            X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                        x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        X1 = X.sel()  # fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
            X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                        x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)

        return X1.mean(x_feature_dim).expand_dims({x_feature_dim: [0], 'ND': [0]})


class BiasCorrectedEnsembleMean:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False,  parallel_in_memory=True):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
            Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        X1 = X.sel()  # fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        Y1 = Y.sel()  # fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        self.regrid_coords_lat = Y1.coords[y_lat_dim].values
        self.regrid_coords_lon = Y1.coords[y_lon_dim].values
        if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
            X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                        x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)

        self.normx = Normal()
        self.normx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        self.normy = Normal()
        self.normy.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        X1 = X.sel()  # fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
            X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim,
                        x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)

        X1 = self.normx.transform(
            X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        X1 = X1.mean(x_feature_dim).expand_dims({x_feature_dim: [0]})
        return self.normy.inverse_transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim).expand_dims({'ND': [0]})


class rMultipleLinearRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = LinearRegression


class rPoissonRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = PoissonRegressionOne


class rGammaRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = GammaRegressionOne



class rEinsteinLearningMachine(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = EinsteinLearningMachine

class rMultiLayerPerceptron(BaseEstimator):
    def __init__(self, hidden_layer_sizes=None, **kwargs):
        if hidden_layer_sizes is not None:
            kwargs['hidden_layer_sizes'] = hidden_layer_sizes
        else:
            kwargs['hidden_layer_sizes'] = (5,)
        super().__init__(**kwargs)
        self.model_type = MLPRegressor


class rRandomForest(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = RandomForestRegressorOne


class rRidgeRegression(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = Ridge


class rExtremeLearningMachine(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = ELMRegressor
