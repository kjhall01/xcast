import cv2
import numpy as np
from scipy.interpolate import interp2d
from sklearn.decomposition import PCA, IncrementalPCA
import xarray as xr
import numpy as np
import dask.array as da
import uuid
import h5py
from pathlib import Path
from ..core.utilities import *
from ..core.progressbar import *
from .missing_values import *


def regrid(X, lons, lats, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', use_dask=True, feat_chunks=1, samp_chunks=1, destination='.xcast_worker_space' ):
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)

	same = True
	if len(X.coords[x_lat_dim].values) == len(lats):
		for i in range(len(X.coords[x_lat_dim].values)):
			if X.coords[x_lat_dim].values[i] != lats[i]:
				same=False
	else:
		same = False
	if len(X.coords[x_lon_dim].values) == len(lons):
		for i in range(len(X.coords[x_lon_dim].values)):
			if X.coords[x_lon_dim].values[i] != lons[i]:
				same=False
	else:
		same = False
	if same:
		return X

	X1 = fill_space_mean(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim )
	X1 = X1.chunk({x_feature_dim: max(X.shape[list(X.dims).index(x_feature_dim)] // feat_chunks, 1), x_sample_dim: max(X.shape[list(X.dims).index(x_sample_dim)] // samp_chunks, 1) }).transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)

	if use_dask:
		id =  Path(destination) / str(uuid.uuid4())
		hdf = h5py.File(id, 'w')
	else:
		hdf = None

	results, seldct = [], {}
	feature_ndx = 0
	for i in range(len(X1.chunks[list(X1.dims).index(x_feature_dim)])):
		sample_ndx = 0
		results.append([])
		for j in range(len(X1.chunks[list(X1.dims).index(x_sample_dim)])):
			x_isel = {x_feature_dim: slice(feature_ndx, feature_ndx + X1.chunks[list(X1.dims).index(x_feature_dim)][i]), x_sample_dim: slice(sample_ndx, sample_ndx + X1.chunks[list(X1.dims).index(x_sample_dim)][j])}
			results[i].append(regrid_chunk(X1.isel(**x_isel), lats, lons,  x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , use_dask=use_dask, hdf=hdf, feature_ndx=feature_ndx, sample_ndx=sample_ndx))
			sample_ndx += X1.chunks[list(X1.dims).index(x_sample_dim)][j]
		feature_ndx += X1.chunks[list(X1.dims).index(x_feature_dim)][i]
		if not use_dask:
			results[i] = np.concatenate(results[i], axis=1)
	if not use_dask:
		results = np.concatenate(results, axis=0)
	else:
		results = []
		hdf.close()
		hdf = h5py.File(id, 'r')
		feature_ndx = 0
		for i in range(len(X1.chunks[list(X1.dims).index(x_feature_dim)])):
			sample_ndx = 0
			results.append([])
			for j in range(len(X1.chunks[list(X1.dims).index(x_sample_dim)])):
				results[i].append(da.from_array(hdf['data_{}_{}'.format(feature_ndx, sample_ndx)]))
				sample_ndx += X1.chunks[list(X1.dims).index(x_sample_dim)][j]
			results[i] = da.concatenate(results[i], axis=1)
			feature_ndx += X1.chunks[list(X1.dims).index(x_feature_dim)][i]
		results = da.concatenate(results, axis=0)
	X1 = X1.transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)
	coords = {
		x_lat_dim: lats,
		x_lon_dim: lons,
		x_feature_dim: X1.coords[x_feature_dim].values,
		x_sample_dim: X1.coords[x_sample_dim].values
	}
	ret = xr.DataArray(data=results, coords=coords, dims=X1.dims)
	selection={x_lat_dim:lats, x_lon_dim:lons}
	mask = X.sel(**selection, method='nearest') / X.sel(**selection, method='nearest')
	mask.coords[x_lat_dim] = lats
	mask.coords[x_lon_dim] = lons
	return ret * mask


def regrid_chunk(X, lats, lons, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', use_dask=False, hdf=None, feature_ndx=0, sample_ndx=0 ):
	res = []
	#X1 = fill_space_mean(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
	data = X.values
	for i in range(data.shape[0]):
		res.append([])
		for j in range(data.shape[1]):
			interp_func = interp2d(X.coords[x_lon_dim].values, X.coords[x_lat_dim].values, data[i,j, :,:], kind='linear')
			interped = interp_func(lons, lats)
			res[i].append(interped)
	res = np.asarray(res)
	if use_dask:
		hdf.create_dataset('data_{}_{}'.format(feature_ndx, sample_ndx), data=res)
		return 'data_{}_{}'.format(feature_ndx, sample_ndx)
	else:
		return res


class SpatialPCA:
	def __init__(self, use_dask=True, **kwargs):
		self.use_dask = use_dask
		self.model_type = IncrementalPCA
		self.models = None
		self.kwargs = kwargs
		self.shape = {'LATITUDE':0, 'LONGITUDE':0, 'FIT_SAMPLES': 0, 'INPUT_FEATURES':0}
		self.count, self.total = 0, 1
		self.transform_lats = None
		self.transform_lons = None

	def fit(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', feat_chunks=1, samp_chunks=1, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self._save_data_shape(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self._gen_models()
		self.nanmask = (X / X).mean(x_sample_dim)

		X1 = X.transpose( x_feature_dim, x_sample_dim,  x_lat_dim, x_lon_dim)
		X1 = self._chunk(X1, x_lat_dim, x_lon_dim, samp_chunks, feat_chunks)
		X1 = fill_space_mean(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim )

		self.lat_dim, self.lon_dim = x_lat_dim, x_lon_dim
		self.sample_dim, self.feature_dim = x_sample_dim, x_feature_dim

		if verbose:
			self.prog = ProgressBar(self.total, label='Fitting Spatial PCA {}:', step=1)
			self.prog.show(self.count)

		feat_ndx_low = 0
		for i in range(len(self.feat_chunks)):
			samp_ndx_low = 0
			for j in range(len(self.samp_chunks)):
				x_isel = {x_feature_dim: slice(feat_ndx_low, feat_ndx_low + self.feat_chunks[i]), x_sample_dim: slice(samp_ndx_low, samp_ndx_low + self.samp_chunks[j])}
				chunk_of_x = X1.isel(**x_isel)
				self._apply_fit_to_chunk(chunk_of_x, feat_ndx_low, samp_ndx_low, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, verbose=verbose )
				samp_ndx_low += self.samp_chunks[j]
			feat_ndx_low += self.feat_chunks[i]

		if verbose:
			self.prog.finish()
			self.count = 0

	def _apply_fit_to_chunk(self, X1, feat_ndx_low, samp_ndx_low, x_lat_dim='Y', x_lon_dim='X', verbose=False):
		x_data = X1.values
		for i in range(x_data.shape[0]):
			x_train = x_data[i, :, :, :]
			if len(x_train.shape) < 3:
				x_train = np.expand_dims(x_train, 0)
			x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
			self.models[i + feat_ndx_low].partial_fit(x_train)
			self.count += 1
			if verbose:
				self.prog.show(self.count)

	def transform(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', feat_chunks=1, samp_chunks=1 , verbose=False, override=False, destination='.xcast_worker_space'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self._check_m_compatibility(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = X.transpose( x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)
		X1 = X1.chunk({x_feature_dim: 1, x_sample_dim: max(self.shape['FIT_SAMPLES'] // len(self.samp_chunks), 1)})

		if self.use_dask:
			id =  Path(destination) / str(uuid.uuid4())
			self.hdf5 = h5py.File(id, 'w')
		if verbose:
			self.prog = ProgressBar(self.total, label='SPATIAL PCA TRANSFORM:', step=10)
			self.prog.show(self.count)

		if override or (self.transform_lats is None) or (self.transform_lons is None ):
			self.transform_lats = X1.coords[x_lat_dim].values
			self.transform_lons = X1.coords[x_lon_dim].values

		results = []
		feat_ndx_low = 0
		for i in range(len(self.feat_chunks)):
			results.append([])
			samp_ndx_low = 0
			for j in range(len(self.samp_chunks)):
				x_isel = {x_feature_dim: slice(feat_ndx_low, feat_ndx_low + self.feat_chunks[i]), x_sample_dim: slice(samp_ndx_low, samp_ndx_low + self.samp_chunks[j])}
				chunk_of_x = X1.isel(**x_isel)
				res = self._apply_transform_to_chunk(chunk_of_x, feat_ndx_low, samp_ndx_low, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, verbose=verbose )
				results[i].append(res)
				samp_ndx_low += self.samp_chunks[j]
			feat_ndx_low += self.feat_chunks[i]
			if not self.use_dask:
				results[i] = np.concatenate(results[i], axis=1)
		if not self.use_dask:
			results = np.concatenate(results, axis=0)
		else:
			results = []
			self.hdf5.close()
			self.hdf5 = h5py.File(id, 'r')
			feat_ndx_low = 0
			for i in range(len(self.feat_chunks)):
				samp_ndx_low = 0
				results.append([])
				for j in range(len(self.samp_chunks)):
					results[i].append(da.from_array(self.hdf5['data_{}_{}'.format(feat_ndx_low, samp_ndx_low)]))
					samp_ndx_low += self.samp_chunks[j]
				results[i] = da.concatenate(results[i], axis=0)
				feat_ndx_low += self.feat_chunks[i]
			results = da.concatenate(results, axis=1)

		if len(results.shape) < 4 and self.shape['INPUT_FEATURES'] == 1:
			if self.use_dask:
				results = da.expand_dims(results, axis=0)
			else:
				results = np.expand_dims(results, axis=0)

		if len(results.shape) < 4 and self.shape['FIT_SAMPLES'] == 1:
			if self.use_dask:
				results = da.expand_dims(results, axis=1)
			else:
				results = np.expand_dims(results, axis=1)
		if verbose:
			self.prog.finish()
			self.count = 0

		coords = {
			self.lat_dim: [i for i in range(results.shape[-2])],
			self.lon_dim: [i for i in range(results.shape[-1])],
			self.sample_dim: X1.coords[x_sample_dim].values,
			self.feature_dim: X1.coords[x_feature_dim].values
		}
		dims = [ self.feature_dim, self.sample_dim, self.lat_dim, self.lon_dim]
		return xr.DataArray(data=results, coords=coords, dims=dims)


	def _apply_transform_to_chunk(self, X1, feat_ndx_low, samp_ndx_low, x_lat_dim='Y', x_lon_dim='X', verbose=False):
		x_data = X1.values
		results = []
		for i in range(x_data.shape[0]):
			results.append([])
			for j in range(x_data.shape[1]):
				x_train = x_data[i, j, :, :]

				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				x_train = x_train.reshape(1, x_train.shape[0]*x_train.shape[1])
				res = np.sum(self.models[i + feat_ndx_low].components_ * x_train, axis=0).reshape(x_data.shape[3], x_data.shape[2])
				if len(res.shape) < 2:
					res = np.expand_dims(res, 0)
				results[i].append(res)
				self.count += 1
				if verbose:
					self.prog.show(self.count)
		results = np.asarray(results)

		if len(results.shape) < 4 and x_data.shape[0] == 1:
			results = np.expand_dims(results, 0)
		if len(results.shape) < 4 and x_data.shape[1] == 1:
			results = np.expand_dims(results, 1)

		assert len(results.shape) == 4, 'Must be Feat x Samp x Mode x Fake Long'

		if self.use_dask:
			self.hdf5.create_dataset('data_{}_{}'.format(feat_ndx_low, samp_ndx_low), data=results)
			return 'data_{}_{}'.format(feat_ndx_low, samp_ndx_low)
		else:
			return results

	def _chunk(self, X, x_feature_dim, x_sample_dim, features_chunks, samples_chunks):
		X1 = X.chunk({x_feature_dim: 1 , x_sample_dim: max(self.shape['FIT_SAMPLES'] // samples_chunks, 1)})
		self.feat_chunks = X1.chunks[0]
		self.samp_chunks = X1.chunks[1]
		return X1

	def _gen_models(self):
		self.models = []
		for j in range(self.shape['INPUT_FEATURES']):
			self.models.append(self.model_type(**self.kwargs))


	def _save_data_shape(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		self.shape['LATITUDE'] =  X.shape[list(X.dims).index(x_lat_dim)]
		self.shape['LONGITUDE'] =  X.shape[list(X.dims).index(x_lon_dim)]
		self.shape['FIT_SAMPLES'] =  X.shape[list(X.dims).index(x_sample_dim)]
		self.shape['INPUT_FEATURES'] =  X.shape[list(X.dims).index(x_feature_dim)]
		self.total = self.shape['INPUT_FEATURES'] * self.shape['FIT_SAMPLES']

	def _check_m_compatibility(self, X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
		assert X.shape[list(X.dims).index(x_feature_dim)] == self.shape['INPUT_FEATURES'], 'BaseMME.predict requires X-Predict to have the same Feature size as X-Train'

	def eofs(self):
		modes = self.models[0].components_.shape[0]
		eofs = np.asarray([self.models[i].components_ for i in range(len(self.models))]).reshape(len(self.models), modes, self.shape['LATITUDE'], self.shape['LONGITUDE'])
		coords = {
			self.feature_dim: self.nanmask.coords[self.feature_dim].values,
			'MODE': [i for i in range(modes)],
			self.lat_dim: self.nanmask.coords[self.lat_dim].values,
			self.lon_dim: self.nanmask.coords[self.lon_dim].values,
		}
		ret = xr.DataArray(data=eofs, coords=coords, dims=[self.feature_dim, 'MODE', self.lat_dim, self.lon_dim ])
		return ret * self.nanmask






def gaussian_smooth(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, feature_chunks=1, sample_chunks=1, destination='.xcast_worker_space' ):
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	#X1 = fill_space_mean(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim )
	X1 = X.chunk({x_feature_dim: max(X.shape[list(X.dims).index(x_feature_dim)] // feature_chunks, 1), x_sample_dim: max(X.shape[list(X.dims).index(x_sample_dim)] // sample_chunks, 1) }).transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)

	if use_dask:
		id =  Path(destination) / str(uuid.uuid4())
		hdf = h5py.File(id, 'w')
	else:
		hdf = None

	results, seldct = [], {}
	feature_ndx = 0
	for i in range(len(X1.chunks[list(X1.dims).index(x_feature_dim)])):
		sample_ndx = 0
		results.append([])
		for j in range(len(X1.chunks[list(X1.dims).index(x_sample_dim)])):
			x_isel = {x_feature_dim: slice(feature_ndx, feature_ndx + X1.chunks[list(X1.dims).index(x_feature_dim)][i]), x_sample_dim: slice(sample_ndx, sample_ndx + X1.chunks[list(X1.dims).index(x_sample_dim)][j])}
			results[i].append(gaussian_smooth_chunk(X1.isel(**x_isel), feature_ndx, sample_ndx, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , use_dask=use_dask, kernel=kernel, hdf=hdf))
			sample_ndx += X1.chunks[list(X1.dims).index(x_sample_dim)][j]
		feature_ndx += X1.chunks[list(X1.dims).index(x_feature_dim)][i]
		if not use_dask:
			results[i] = np.concatenate(results[i], axis=1)
	if not use_dask:
		results = np.concatenate(results, axis=0)
	else:
		results = []
		hdf.close()
		hdf = h5py.File(id, 'r')
		feature_ndx = 0
		for i in range(len(X1.chunks[list(X1.dims).index(x_feature_dim)])):
			sample_ndx = 0
			results.append([])
			for j in range(len(X1.chunks[list(X1.dims).index(x_sample_dim)])):
				results[i].append(da.from_array(hdf['data_{}_{}'.format(feature_ndx, sample_ndx)]))
				sample_ndx += X1.chunks[list(X1.dims).index(x_sample_dim)][j]
			results[i] = da.concatenate(results[i], axis=0)
			feature_ndx += X1.chunks[list(X1.dims).index(x_feature_dim)][i]
		results = da.concatenate(results, axis=1)
	#X1 = X1.transpose(x_sample_dim, x_feature_dim, x_lat_dim, x_lon_dim)
	return xr.DataArray(data=results, coords=X1.coords, dims=X1.dims)


def gaussian_smooth_chunk(X, feature_ndx, sample_ndx, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, hdf=None ):
	res = []
	data = X.values
	for i in range(data.shape[0]):
		res.append([])
		for j in range(data.shape[1]):
			toblur = data[i, j, :, :]
			mask = np.isnan(toblur)
			toblur2 = toblur.copy()
			toblur2[mask] = np.nanmean(toblur)
			blurred = cv2.GaussianBlur(toblur2, kernel,0)
			blurred[mask] = np.nan
			res[i].append(blurred)
	res = np.asarray(res)
	if use_dask:
		hdf.create_dataset('data_{}_{}'.format(feature_ndx, sample_ndx), data=res)
		return 'data_{}_{}'.format(feature_ndx, sample_ndx)
	else:
		return res
