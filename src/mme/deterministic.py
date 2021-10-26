import numpy as np
from sklearn.linear_model import LinearRegression, GammaRegressor, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from ..flat_estimators.regressors import *
from ..regression.base_regressor import *
from ..preprocessing import *

class mEnsembleMean:
	def __init__(self, **kwargs):
		pass

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False,  parallel_in_memory=True ):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)

		return X1.mean(x_feature_dim).expand_dims({x_feature_dim:[0], 'ND':[0]})


class mBiasCorrectedEnsembleMean:
	def __init__(self, **kwargs):
		pass

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.normx = Normal()
		self.normx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.normy = Normal()
		self.normy.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.normx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = X1.mean(x_feature_dim).expand_dims({x_feature_dim:[0]})
		return self.normy.inverse_transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim).expand_dims({'ND':[0]})



class mMultipleRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = LinearRegression

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False,  parallel_in_memory=True ):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.mm = MinMax()
		self.mm.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)

		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks,  verbose=verbose, parallel_in_memory=parallel_in_memory)


class mPoissonRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = PoissonRegressionOne

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.mm = MinMax(min=0.00000001, max=2)
		self.mm.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)

class mGammaRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = GammaRegressionOne

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)



		self.mm = MinMax(min=0.00000001, max=2)
		self.mm.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		X1 = X1.where(X1 > 0, other=0.00000001)
		Y1 = Y1.where(Y1 > 0, other=0.00000001)

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = X1.where(X1 > 0, other=0.00000001)
		return super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)


class mPrincipalComponentsRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = LinearRegression

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, n_components=2 ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.mm = MinMax()
		self.mm.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.pca = PrincipalComponentsAnalysis(n_components=n_components)
		self.pca.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mm.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)



class mMultiLayerPerceptron(BaseRegressor):
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		super().__init__(**kwargs)
		self.model_type = MLPRegressor

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True ):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.normy = Normal()
		self.normy.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.normy.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.

		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		preds =  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)
		return self.normy.inverse_transform(preds, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)


class mRandomForest(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = RandomForestRegressor

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.normy = Normal()
		self.normy.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.normy.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1,  samp_chunks=1, verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		preds =  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)
		return self.normy.inverse_transform(preds, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)


class mRidgeRegressor(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = Ridge

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)


class mExtremeLearningMachine(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ELMRegressor

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)

		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.normy = Normal()
		self.normy.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.normy.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1,  verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		preds =  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks,  samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)
		return self.normy.inverse_transform(preds, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)


class mExtremeLearningMachinePCA(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ELMRegressor

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False ,  n_components=2, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.pca = PrincipalComponentsAnalysis(n_components=n_components)
		self.pca.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.normy = Normal()
		self.normy.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.normy.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose, parallel_in_memory=parallel_in_memory)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 ,  feat_chunks=1, samp_chunks=1, verbose=False, parallel_in_memory=True):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, use_dask=not parallel_in_memory)


		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		preds =  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose, parallel_in_memory=parallel_in_memory)
		return self.normy.inverse_transform(preds, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
