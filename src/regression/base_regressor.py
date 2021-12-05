import xarray as xr
import numpy as np
import dask.array as da
import uuid, sys
import os
from ..core.utilities import *
from ..core.progressbar import *
from ..flat_estimators.regressors import *
from dask.distributed import Client, progress, as_completed
import dask.array as da
import dask
import dask.diagnostics as dd
from sklearn.linear_model import LinearRegression


def apply_fit_to_block(x_data, y_data, mme=LinearRegression, ND=1, kwargs={}):
	models = []
	for i in range(x_data.shape[0]):
		models.append([])
		for j in range(x_data.shape[1]):
			models[i].append([])
			x_train = x_data[i, j, :, :]
			y_train = y_data[i, j, :, :]
			if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
				temp_mme = NanRegression
			else:
				temp_mme = mme
			if len(x_train.shape) < 2:
				x_train = x_train.reshape(-1,1)
			if len(y_train.shape) < 2:
				y_train = y_train.reshape(-1,1)
			empty = np.empty(ND, object)
			for k in range(ND):
				models[i][j].append(temp_mme(**kwargs))
				models[i][j][k].fit(x_train, y_train)
				empty[k] = models[i][j][k]
			models[i][j] = empty
	models = np.array(models, dtype=np.dtype('O'))
	return models

def apply_predict_to_block(x_data, models):
	ret = []
	for i in range(x_data.shape[0]):
		ret.append([])
		for j in range(x_data.shape[1]):
			ret[i].append([])
			x_train = x_data[i, j, :, :]
			if len(x_train.shape) < 2:
				x_train = x_train.reshape(-1,1)
			for k in range(models.shape[2]):
				preds = models[i][j][k].predict(x_train)
				if len(preds.shape) < 2:
					preds = np.expand_dims(preds, axis=1)
				ret[i][j].append(preds)
	return np.asarray(ret)

class BaseRegressor:
	""" Base MME class
	implements .fit(X, Y) and .predict(X)
	can be sub-classed to extend to new statistical methods
	new methods must implement .fit(x, y) and .predict(x)
	and then sub-class's .model_type must be set to the constructor of the new method """

	def __init__(self, client=None, ND=1, **kwargs):
		self.model_type = LinearRegression
		self.models = None
		self.ND = ND
		self.client = client
		self.kwargs = kwargs
		self.shape = {'LATITUDE':0, 'LONGITUDE':0, 'FIT_SAMPLES': 0, 'INPUT_FEATURES':0, 'OUTPUT_FEATURES': 0}
		self.count, self.total = 0, 1

	def fit(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False, parallel_in_memory=True, rechunk=True ):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self._check_xyt_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim, y_lat_dim, y_lon_dim, y_sample_dim)
		self._save_data_shape(X, Y, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, y_feature_dim)
		X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		if rechunk:
			X1, Y1 = self._chunk(X1, Y1, x_lat_dim, x_lon_dim, y_lat_dim, y_lon_dim, x_feature_dim, y_feature_dim, x_sample_dim, y_sample_dim, lat_chunks, lon_chunks)
		x_data = X1.data
		y_data = Y1.data
		if verbose:
			with dd.ProgressBar():
				self.models = da.map_blocks(apply_fit_to_block, x_data, y_data, drop_axis=[2,3], new_axis=[3], mme=self.model_type, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).compute()
		else:
			self.models = da.map_blocks(apply_fit_to_block, x_data, y_data, drop_axis=[2,3], new_axis=[3], mme=self.model_type, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).compute()
		

	def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, lat_chunks=1, lon_chunks=1 ,  feat_chunks=1, samp_chunks=1, destination='.xcast_worker_space', verbose=False, rechunk=True, parallel_in_memory=True):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self._check_xym_compatibility(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		if rechunk:
			X1 = X.chunk({x_lat_dim: max(self.shape['LATITUDE'] // lat_chunks,1), x_lon_dim: max(self.shape['LONGITUDE'] // lon_chunks,1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.models = da.from_array(self.models, chunks=(max(self.shape['LATITUDE'] // lat_chunks,1), max(self.shape['LONGITUDE'] // lon_chunks,1), self.ND))
		x_data = X1.data
		if verbose:
			with dd.ProgressBar():
				results = da.map_blocks(apply_predict_to_block, x_data, self.models, drop_axis=[2,3], new_axis=[3, 4, 5], dtype=float).compute()
		else:
			results = da.map_blocks(apply_predict_to_block, x_data, self.models, drop_axis=[2,3], new_axis=[3, 4, 5], dtype=float).compute()

		if len(results.shape) < 4 and self.shape['LATITUDE'] == 1:
			results = da.expand_dims(results, axis=0)

		if len(results.shape) < 4 and self.shape['LONGITUDE'] == 1:
			results = da.expand_dims(results, axis=1)

		if verbose and not parallel_in_memory:
			self.prog.finish()
			self.count = 0

		coords = {
			x_lat_dim: X1.coords[x_lat_dim].values,
			x_lon_dim: X1.coords[x_lon_dim].values,
			x_sample_dim: X1.coords[x_sample_dim].values,
			x_feature_dim: [i for i in range(results.shape[-1])],
			'ND': [i for i in range(self.ND)]
		}
		dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
		attrs = X1.attrs 
		attrs.update({'generated by': 'XCAST Regressor {}'.format(self.model_type)})
		return xr.DataArray(data=results, coords=coords, dims=dims, attrs=attrs)


	def _chunk(self, X, Y, x_lat_dim, x_lon_dim, y_lat_dim, y_lon_dim, x_feature_dim, y_feature_dim, x_sample_dim, y_sample_dim, lat_chunks, lon_chunks):
		X1 = X.chunk({x_lat_dim: max(self.shape['LATITUDE'] // lat_chunks,1), x_lon_dim: max(self.shape['LONGITUDE'] // lon_chunks,1), x_feature_dim:self.shape['INPUT_FEATURES'], x_sample_dim:-1})
		Y1 = Y.chunk({y_lat_dim: max(self.shape['LATITUDE'] // lat_chunks, 1), y_lon_dim: max(self.shape['LONGITUDE'] // lon_chunks,1), y_feature_dim:self.shape['OUTPUT_FEATURES'], y_sample_dim:-1})

		self.lat_chunks, self.lon_chunks = X1.chunks[0], X1.chunks[1]
		return X1, Y1

	def _save_data_shape(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None,  y_feature_dim=None ):
		self.shape['LATITUDE'] =  X.shape[list(X.dims).index(x_lat_dim)]
		self.shape['LONGITUDE'] =  X.shape[list(X.dims).index(x_lon_dim)]
		self.shape['FIT_SAMPLES'] =  X.shape[list(X.dims).index(x_sample_dim)]
		self.shape['INPUT_FEATURES'] =  X.shape[list(X.dims).index(x_feature_dim)]
		self.shape['OUTPUT_FEATURES'] =  Y.shape[list(Y.dims).index(y_feature_dim)]
		self.total = self.shape['LATITUDE'] * self.shape['LONGITUDE'] * self.ND

	def _check_xym_compatibility(self, X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
		assert X.shape[list(X.dims).index(x_lat_dim)] == self.shape['LATITUDE'], 'BaseMME.predict requires X-Predict to have the same Latitude size as X-Train'
		assert X.shape[list(X.dims).index(x_lon_dim)] == self.shape['LONGITUDE'], 'BaseMME.predict requires X-Predict to have the same Longitude size as X-Train'
		assert X.shape[list(X.dims).index(x_feature_dim)] == self.shape['INPUT_FEATURES'], 'BaseMME.predict requires X-Predict to have the same Feature size as X-Train'

	def _check_xyt_compatibility(self, X, Y, x_lat_dim, x_lon_dim, x_sample_dim, y_lat_dim, y_lon_dim, y_sample_dim):
		assert X.shape[list(X.dims).index(x_lat_dim)] == Y.shape[list(Y.dims).index(y_lat_dim)], 'BaseMME.fit requires X and Y to have the same Latitude size'
		assert X.shape[list(X.dims).index(x_lon_dim)] == Y.shape[list(Y.dims).index(y_lon_dim)], 'BaseMME.fit requires X and Y to have the same Longitude size'
		assert X.shape[list(X.dims).index(x_sample_dim)] == Y.shape[list(Y.dims).index(y_sample_dim)], 'BaseMME.fit requires X and Y to have the same Sample size'
