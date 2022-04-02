import xarray as xr
import numpy as np
import dask.array as da
import uuid, sys, os
from ..core.utilities import check_all, check_xyt_compatibility, guess_coords, shape
from ..core.chunking import  align_chunks
from ..core.progressbar import ProgressBar
import dask.diagnostics as dd
from ..flat_estimators.classifiers import NanClassifier, RFClassifier, POELMClassifier
from ..flat_estimators.regressors import NanRegression
from sklearn.decomposition import PCA 

def apply_fit_to_block(x_data, y_data, mme=POELMClassifier, ND=1, kwargs={}):
	models = []
	for i in range(x_data.shape[0]):
		models.append([])
		for j in range(x_data.shape[1]):
			models[i].append([])
			x_train = x_data[i, j, :, :]
			y_train = y_data[i, j, :, :]
			if np.isnan(np.min(x_train)) or np.isnan(np.min(y_train)):
				temp_mme = NanClassifier
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

def apply_fit_x_to_block(x_data, mme=PCA, ND=1, kwargs={}):
	models = []
	for i in range(x_data.shape[0]):
		models.append([])
		for j in range(x_data.shape[1]):
			models[i].append([])
			x_train = x_data[i, j, :, :]
			if np.isnan(np.min(x_train)):
				temp_mme = NanClassifier
			else:
				temp_mme = mme
			if len(x_train.shape) < 2:
				x_train = x_train.reshape(-1,1)
			empty = np.empty(ND, object)
			for k in range(ND):
				models[i][j].append(temp_mme(**kwargs))
				models[i][j][k].fit(x_train)
				empty[k] = models[i][j][k]
			models[i][j] = empty
	models = np.array(models, dtype=np.dtype('O'))
	return models

def apply_predict_proba_to_block(x_data, models):
	ret = []
	for i in range(x_data.shape[0]):
		ret.append([])
		for j in range(x_data.shape[1]):
			ret[i].append([])
			x_train = x_data[i, j, :, :]
			if len(x_train.shape) < 2:
				x_train = x_train.reshape(-1,1)
			for k in range(models.shape[2]):
				ret1 = models[i][j][k].predict_proba(x_train)
				ret[i][j].append(ret1)

	return np.asarray(ret)

def apply_transform_to_block(x_data, models):
	ret = []
	for i in range(x_data.shape[0]):
		ret.append([])
		for j in range(x_data.shape[1]):
			ret[i].append([])
			x_train = x_data[i, j, :, :]
			if len(x_train.shape) < 2:
				x_train = x_train.reshape(-1,1)
			for k in range(models.shape[2]):
				ret1 = models[i][j][k].transform(x_train)
				ret[i][j].append(ret1)

	return np.asarray(ret)


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


class BaseEstimator:
	""" BaseEstimator class
	implements .fit(X, Y) and, .predict_proba(X), .predict(X)
	can be sub-classed to extend to new statistical methods
	new methods must implement .fit(x, y) and .predict(x)
	and then sub-class's .model_type must be set to the constructor of the new method """

	def __init__(self, client=None, ND=1, lat_chunks=1, lon_chunks=1, verbose=False, **kwargs):
		self.model_type = POELMClassifier
		self.models, self.ND = None, ND
		self.client, self.kwargs = client, kwargs
		self.verbose=verbose
		self.lat_chunks, self.lon_chunks = lat_chunks, lon_chunks
		self.latitude, self.longitude, self.features = None, None, None

	def fit(self, X, *args,  x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None, rechunk=True ):
		if len(args) > 0:
			assert len(args) < 2, 'too many args'
			Y = args[0]

			x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
			y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
			check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
			check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
			check_xyt_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
			self.latitude, self.longitude, _, self.features = shape(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

			X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
			Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

			if rechunk:
				X1, Y1 = align_chunks(X1, Y1,  self.lat_chunks, self.lon_chunks, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim)

			x_data = X1.data
			y_data = Y1.data
			if not isinstance(x_data, da.core.Array):
				x_data = da.from_array(x_data)
			if not isinstance(y_data, da.core.Array):
				y_data = da.from_array(y_data)
			
			if self.verbose:
				with dd.ProgressBar():
					self.models = da.map_blocks(apply_fit_to_block, x_data, y_data, drop_axis=[2,3], new_axis=[3], mme=self.model_type, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).compute()
			else:
				self.models = da.map_blocks(apply_fit_to_block, x_data, y_data, drop_axis=[2,3], new_axis=[3], mme=self.model_type, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).compute()
			if type(self.models) == np.ndarray:
				self.models = da.from_array(self.models, chunks=(max(self.latitude // self.lat_chunks,1), max(self.longitude // self.lon_chunks,1), self.ND))
		else: 
			x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
			check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
			self.latitude, self.longitude, _, self.features = shape(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

			X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

			x_data = X1.data
			if not isinstance(x_data, da.core.Array):
				x_data = da.from_array(x_data)

			if self.verbose:
				with dd.ProgressBar():
					self.models = da.map_blocks(apply_fit_x_to_block, x_data, drop_axis=[2,3], new_axis=[3], mme=self.model_type, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).compute()
			else:
				self.models = da.map_blocks(apply_fit_x_to_block, x_data, drop_axis=[2,3], new_axis=[3], mme=self.model_type, ND=self.ND, kwargs=self.kwargs, meta=np.array((), dtype=np.dtype('O'))).compute()
			if type(self.models) == np.ndarray:
				self.models = da.from_array(self.models, chunks=(max(self.latitude // self.lat_chunks,1), max(self.longitude // self.lon_chunks,1), self.ND))


	def predict_proba(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True ):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		xlat, xlon, xsamp, xfeat = shape(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

		assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
		assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
		assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'

		if rechunk:
			X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks,1), x_lon_dim: max(xlon // self.lon_chunks,1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		else:
			X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		x_data = X1.data
		if self.verbose:
			with dd.ProgressBar():
				results = da.blockwise(apply_predict_proba_to_block, 'ijnkm', x_data, 'ijkl', self.models, 'ijm', new_axes={'n': self.ND}, dtype=float, concatenate=True).compute()
		else:
			results = da.blockwise(apply_predict_proba_to_block, 'ijnkm', x_data, 'ijkl', self.models, 'ijm', new_axes={'n': self.ND}, dtype=float, concatenate=True).compute()

		coords = {
			x_lat_dim: X1.coords[x_lat_dim].values,
			x_lon_dim: X1.coords[x_lon_dim].values,
			x_sample_dim: X1.coords[x_sample_dim].values,
			x_feature_dim: [i for i in range(results.shape[-1])],
			'ND': [i for i in range(self.ND)]
		}
		
		dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
		attrs = X1.attrs 
		attrs.update({'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
		return xr.DataArray(data=results, coords=coords, dims=dims, attrs=attrs)

	def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True ):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		xlat, xlon, xsamp, xfeat = shape(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

		assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
		assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
		assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'

		if rechunk:
			X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks,1), x_lon_dim: max(xlon // self.lon_chunks,1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		else:
			X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		x_data = X1.data
		if self.verbose:
			with dd.ProgressBar():
				results = da.blockwise(apply_transform_to_block, 'ijnkm', x_data, 'ijkl', self.models, 'ijm', new_axes={'n': self.ND}, dtype=float, concatenate=True).compute()
		else:
			results = da.blockwise(apply_transform_to_block, 'ijnkm', x_data, 'ijkl', self.models, 'ijm', new_axes={'n': self.ND}, dtype=float, concatenate=True).compute()

		coords = {
			x_lat_dim: X1.coords[x_lat_dim].values,
			x_lon_dim: X1.coords[x_lon_dim].values,
			x_sample_dim: X1.coords[x_sample_dim].values,
			x_feature_dim: [i for i in range(results.shape[-1])],
			'ND': [i for i in range(self.ND)]
		}
		
		dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
		attrs = X1.attrs 
		attrs.update({'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
		return xr.DataArray(data=results, coords=coords, dims=dims, attrs=attrs)

	def predict(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, rechunk=True ):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		xlat, xlon, xsamp, xfeat = shape(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

		assert xlat == self.latitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lat mismatch'
		assert xlon == self.longitude, 'XCast Estimators require new predictors to have the same dimensions as the training data- lon mismatch'
		assert xfeat == self.features, 'XCast Estimators require new predictors to have the same dimensions as the training data- feat mismatch'

		if rechunk:
			X1 = X.chunk({x_lat_dim: max(xlat // self.lat_chunks,1), x_lon_dim: max(xlon // self.lon_chunks,1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		else:
			X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		x_data = X1.data
		if self.verbose:
			with dd.ProgressBar():
				results = da.blockwise(apply_predict_to_block, 'ijnkm', x_data, 'ijkl', self.models, 'ijm', new_axes={'n': self.ND}, dtype=float, concatenate=True).compute()
		else:
			results = da.blockwise(apply_predict_to_block, 'ijnkm', x_data, 'ijkl', self.models, 'ijm', new_axes={'n': self.ND}, dtype=float, concatenate=True).compute()

		coords = {
			x_lat_dim: X1.coords[x_lat_dim].values,
			x_lon_dim: X1.coords[x_lon_dim].values,
			x_sample_dim: X1.coords[x_sample_dim].values,
			x_feature_dim: [i for i in range(results.shape[-1])],
			'ND': [i for i in range(self.ND)]
		}
		
		dims = [x_lat_dim, x_lon_dim, 'ND', x_sample_dim, x_feature_dim]
		attrs = X1.attrs 
		attrs.update({'generated_by': 'XCast Classifier - {}'.format(self.model_type)})
		return xr.DataArray(data=results, coords=coords, dims=dims, attrs=attrs)