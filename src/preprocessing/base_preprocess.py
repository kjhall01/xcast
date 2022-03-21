from sklearn.decomposition import PCA
import xarray as xr
import numpy as np
import uuid
from ..flat_estimators.regressors import NanRegression
from ..core.utilities import *
from ..core.progressbar import *

class BasePreprocess:
	def __init__(self,  **kwargs):
		self.model_type = PCA
		self.models = None
		self.kwargs = kwargs
		self.shape = {'LATITUDE':0, 'LONGITUDE':0, 'FIT_SAMPLES': 0, 'INPUT_FEATURES':0, 'OUTPUT_FEATURES': 0}
		self.count, self.total = 0, 1

	def fit(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, lat_chunks=1, lon_chunks=1, verbose=False ):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self._save_data_shape(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.models = []

		X1 = X.chunk({x_lat_dim: max(self.shape['LATITUDE'] // lat_chunks, 1), x_lon_dim: max(self.shape['LONGITUDE'] // lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.lat_chunks = X1.chunks[0]
		self.lon_chunks = X1.chunks[1]
		if verbose:
			self.prog = ProgressBar(self.total, label='Fitting {}:'.format(self.model_type.__name__), step=10)
			self.prog.show(self.count)

		lat_ndx_low = 0
		for i in range(len(self.lat_chunks)):
			lon_ndx_low = 0
			for j in range(len(self.lon_chunks)):
				x_isel = {x_lat_dim: slice(lat_ndx_low, lat_ndx_low + self.lat_chunks[i]), x_lon_dim: slice(lon_ndx_low, lon_ndx_low + self.lon_chunks[j])}
				chunk_of_x = X1.isel(**x_isel)
				self._apply_fit_to_chunk(chunk_of_x, lat_ndx_low, lon_ndx_low, verbose=verbose )
				lon_ndx_low += self.lon_chunks[j]
			lat_ndx_low += self.lat_chunks[i]

		if verbose:
			self.prog.finish()
			self.count = 0

	def _apply_fit_to_chunk(self, X1, lat_ndx_low, lon_ndx_low, verbose=False):
		x_data = X1.values
		for i in range(x_data.shape[0]):
			self.models.append([])
			for j in range(x_data.shape[1]):
				x_train = x_data[i, j, :, :]
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				if np.isnan(np.min(x_train)):
					self.models[-1].append(NanRegression(**self.kwargs))
				else:
					self.models[-1].append(self.model_type(**self.kwargs))
				self.models[-1][-1].fit(x_train)
				self.count += 1
				if verbose:
					self.prog.show(self.count)

	def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, lat_chunks=1, lon_chunks=1 , verbose=False):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self._check_xym_compatibility(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = X.chunk({x_lat_dim: max(self.shape['LATITUDE'] // lat_chunks, 1), x_lon_dim: max(self.shape['LONGITUDE'] // lon_chunks, 1)}).transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)


		if verbose:
			self.prog = ProgressBar(self.total, label='Predicting {}:'.format(self.model_type.__name__), step=10)
			self.prog.show(self.count)

		results = []
		lat_ndx_low = 0
		for i in range(len(self.lat_chunks)):
			results.append([])
			lon_ndx_low = 0
			for j in range(len(self.lon_chunks)):
				x_isel = {x_lat_dim: slice(lat_ndx_low, lat_ndx_low + self.lat_chunks[i]), x_lon_dim: slice(lon_ndx_low, lon_ndx_low + self.lon_chunks[j])}
				chunk_of_x = X1.isel(**x_isel)
				res = self._apply_transform_to_chunk(chunk_of_x, lat_ndx_low, lon_ndx_low, verbose=verbose )
				results[i].append(res)
				lon_ndx_low += self.lon_chunks[j]
			lat_ndx_low += self.lat_chunks[i]
			results[i] = np.concatenate(results[i], axis=1)
		results = np.concatenate(results, axis=0)
		

		if len(results.shape) < 4 and self.shape['LATITUDE'] == 1:
			results = np.expand_dims(results, axis=0)

		if len(results.shape) < 4 and self.shape['LONGITUDE'] == 1:
			results = np.expand_dims(results, axis=1)

		if verbose:
			self.prog.finish()
			self.count = 0

		coords = {
			x_lat_dim: X1.coords[x_lat_dim].values,
			x_lon_dim: X1.coords[x_lon_dim].values,
			x_sample_dim: X1.coords[x_sample_dim].values,
			x_feature_dim: [i for i in range(results.shape[-1])]
		}
		attrs = X1.attrs 
		attrs.update({'generated_by': attrs['generated_by'] + '\n  XCAST preprocessing: {}'.format(self.model_type) if 'generated_by' in attrs.keys() else '\n  XCAST preprocessing: {}'.format(self.model_type)})
		dims = [x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim]
		return xr.DataArray(data=results, coords=coords, dims=dims)


	def _apply_transform_to_chunk(self, X1, lat_ndx_low, lon_ndx_low, verbose=False):
		x_data = X1.values
		results = []
		for i in range(x_data.shape[0]):
			results.append([])
			for j in range(x_data.shape[1]):
				x_train = x_data[i, j, :, :]
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				res = self.models[i + lat_ndx_low][j+lon_ndx_low].transform(x_train)
				if len(res.shape) < 2:
					res = np.expand_dims[0]
				results[i].append(res)
				self.count += 1
				if verbose:
					self.prog.show(self.count)
		return results

	def _save_data_shape(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
		self.shape['LATITUDE'] =  X.shape[list(X.dims).index(x_lat_dim)]
		self.shape['LONGITUDE'] =  X.shape[list(X.dims).index(x_lon_dim)]
		self.shape['FIT_SAMPLES'] =  X.shape[list(X.dims).index(x_sample_dim)]
		self.shape['INPUT_FEATURES'] =  X.shape[list(X.dims).index(x_feature_dim)]
		self.total = self.shape['LATITUDE'] * self.shape['LONGITUDE']

	def _check_xym_compatibility(self, X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
		assert X.shape[list(X.dims).index(x_lat_dim)] == self.shape['LATITUDE'], 'BaseMME.predict requires X-Predict to have the same Latitude size as X-Train'
		assert X.shape[list(X.dims).index(x_lon_dim)] == self.shape['LONGITUDE'], 'BaseMME.predict requires X-Predict to have the same Longitude size as X-Train'
		assert X.shape[list(X.dims).index(x_feature_dim)] == self.shape['INPUT_FEATURES'], 'BaseMME.predict requires X-Predict to have the same Feature size as X-Train'
