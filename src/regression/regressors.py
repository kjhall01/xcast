import numpy as np
from sklearn.linear_model import LinearRegression, GammaRegressor, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from ..flat_estimators.regressors import *
from .base_regressor import *
from ..preprocessing import *
#import extremelearning as elm


class rMultipleLinearRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = LinearRegression

class rPoissonRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = PoissonRegressionOne

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
#		X1 = X1.where(X1 > 0, other=0.00000001)
#		Y1 = Y1.where(Y1 > 0, other=0.00000001)
		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
#		X1 = X1.where(X1 > 0, other=0.00000001)
		return super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)


class rGammaRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = GammaRegressionOne

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
#		X1 = X1.where(X1 > 0, other=0.00000001)
#		Y1 = Y1.where(Y1 > 0, other=0.00000001)

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
#		X1 = X1.where(X1 > 0, other=0.00000001)
		return super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)


class rMultiLayerPerceptron(BaseRegressor):
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		super().__init__(**kwargs)
		self.model_type = MLPRegressor

class rRandomForest(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = RandomForestRegressor

class rRidgeRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = Ridge

class rExtremeLearningMachine(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ELMRegressor
