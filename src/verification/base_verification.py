from sklearn.metrics import mean_squared_error
import xarray as xr
import numpy as np
import dask.array as da
import uuid
import h5py
from ..core.utilities import *
from ..core.progressbar import *

class BaseMetric:
	def __init__(self, use_dask=False, **kwargs):
		self.use_dask=use_dask
		self.model_type = mean_squared_error
		self.models = None
		self.kwargs = kwargs
		self.shape = {'LATITUDE':0, 'LONGITUDE':0, 'FIT_SAMPLES': 0, 'INPUT_FEATURES':0, 'OUTPUT_FEATURES': 0}
		self.count, self.total = 0, 1

	def __call__(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self._check_xytm_compatibility(X, Y, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim,  y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self._save_data_shape(X, Y, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, y_feature_dim)

		X1, Y1 = self._chunk(X, Y, x_lat_dim, x_lon_dim, y_lat_dim, y_lon_dim, lat_chunks, lon_chunks)
		X1 = X1.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y1.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

		if verbose:
			self.prog = ProgressBar(self.total, label='Fitting {}:'.format(self.model_type.__name__), step=10)
			self.prog.show(self.count)

		results = []
		lat_ndx_low = 0
		for i in range(len(self.lat_chunks)):
			results.append([])
			lon_ndx_low = 0
			for j in range(len(self.lon_chunks)):
				x_isel = {x_lat_dim: slice(lat_ndx_low, lat_ndx_low + self.lat_chunks[i]), x_lon_dim: slice(lon_ndx_low, lon_ndx_low + self.lon_chunks[j])}
				y_isel = {y_lat_dim: slice(lat_ndx_low, lat_ndx_low + self.lat_chunks[i]), y_lon_dim: slice(lon_ndx_low, lon_ndx_low + self.lon_chunks[j])}
				chunk_of_y = Y1.isel(**x_isel)
				chunk_of_x = X1.isel(**x_isel)
				res = self._apply_function_to_chunk(chunk_of_x, chunk_of_y, lat_ndx_low, lon_ndx_low, verbose=verbose )
				results[i].append(res)
				lon_ndx_low += self.lon_chunks[j]
			lat_ndx_low += self.lat_chunks[i]
			if not self.use_dask:
				results[i] = np.concatenate(results[i], axis=1)
		if not self.use_dask:
			results = np.concatenate(results, axis=0)
		else:
			results = []
			self.hdf5.close()
			self.hdf5 = h5py.File(id, 'r')
			lat_ndx_low = 0
			for i in range(len(self.lat_chunks)):
				lon_ndx_low = 0
				results.append([])
				for j in range(len(self.lon_chunks)):
					results[i].append(da.from_array(self.hdf5['data_{}_{}'.format(lat_ndx_low, lon_ndx_low)]))
					lon_ndx_low += self.lon_chunks[j]
				results[i] = da.concatenate(results[i], axis=1)
				lat_ndx_low += self.lat_chunks[i]
			results = da.concatenate(results, axis=0)

		if len(results.shape) < 4 and self.shape['LATITUDE'] == 1:
			if self.use_dask:
				results = da.expand_dims(results, axis=0)
			else:
				results = np.expand_dims(results, axis=0)

		if len(results.shape) < 4 and self.shape['LONGITUDE'] == 1:
			if self.use_dask:
				results = da.expand_dims(results, axis=1)
			else:
				results = np.expand_dims(results, axis=1)

		if verbose:
			self.prog.finish()
			self.count = 0

		coords = {
			x_lat_dim: X1.coords[x_lat_dim].values,
			x_lon_dim: X1.coords[x_lon_dim].values,
			x_sample_dim: [0],
			x_feature_dim: [i for i in range(results.shape[-1])]
		}
		dims = [x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim]
		return xr.DataArray(data=results, coords=coords, dims=dims)

	def _apply_function_to_chunk(self, X1, Y1, lat_ndx_low, lon_ndx_low, verbose=False):
		x_data, y_data = X1.values, Y1.values
		results = []
		for i in range(x_data.shape[0]):
			results.append([])
			for j in range(x_data.shape[1]):
				x_train = x_data[i, j, :, :]
				y_train = y_data[i, j, :, :]
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				if len(y_train.shape) < 2:
					y_train = y_train.reshape(-1,1)
				res = self.function(x_train, y_train)
				if len(res.shape) < 2:
					res = np.expand_dims(res, 0)
				results[i].append(res)
				self.count += 1
				if verbose:
					self.prog.show(self.count)
		if x_data.shape[0] == 1 and self.shape['LATITUDE'] == 1:
			results = np.expand_dims(results, 0)
		if x_data.shape[1] == 1 and self.shape['LONGITUDE'] == 1:
			results = np.expand_dims(results, 1)
		if self.use_dask:
			self.hdf5.create_dataset('data_{}_{}'.format(lat_ndx_low, lon_ndx_low), data=results)
			return 'data_{}_{}'.format(lat_ndx_low, lon_ndx_low)
		else:
			return results


	def _chunk(self, X, Y, x_lat_dim, x_lon_dim, y_lat_dim, y_lon_dim, lat_chunks, lon_chunks):
		X1 = X.chunk({x_lat_dim: self.shape['LATITUDE'] // lat_chunks, x_lon_dim: self.shape['LONGITUDE'] // lon_chunks})
		Y1 = Y.chunk({y_lat_dim: self.shape['LATITUDE'] // lat_chunks, y_lon_dim: self.shape['LONGITUDE'] // lon_chunks})

		self.lat_chunks = X1.chunks[0]
		self.lon_chunks = X1.chunks[1]
		return X1, Y1

	def _save_data_shape(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  y_feature_dim='M' ):
		self.shape['LATITUDE'] =  X.shape[list(X.dims).index(x_lat_dim)]
		self.shape['LONGITUDE'] =  X.shape[list(X.dims).index(x_lon_dim)]
		self.shape['FIT_SAMPLES'] =  X.shape[list(X.dims).index(x_sample_dim)]
		self.shape['INPUT_FEATURES'] =  X.shape[list(X.dims).index(x_feature_dim)]
		self.shape['OUTPUT_FEATURES'] =  Y.shape[list(Y.dims).index(y_feature_dim)]
		self.total = self.shape['LATITUDE'] * self.shape['LONGITUDE']

	def _check_xytm_compatibility(self, X, Y, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim):
		assert X.shape[list(X.dims).index(x_lat_dim)] == Y.shape[list(Y.dims).index(y_lat_dim)], 'BaseMME.fit requires X and Y to have the same Latitude size'
		assert X.shape[list(X.dims).index(x_lon_dim)] == Y.shape[list(Y.dims).index(y_lon_dim)], 'BaseMME.fit requires X and Y to have the same Longitude size'
		assert X.shape[list(X.dims).index(x_sample_dim)] == Y.shape[list(Y.dims).index(y_sample_dim)], 'BaseMME.fit requires X and Y to have the same Sample size'
		assert X.shape[list(X.dims).index(x_feature_dim)] == Y.shape[list(Y.dims).index(y_feature_dim)], 'BaseMME.predict requires X-Predict to have the same Feature size as X-Train'
