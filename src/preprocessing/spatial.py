import cv2
import numpy as np
from scipy.interpolate import interp2d
from sklearn.decomposition import PCA, IncrementalPCA
import xarray as xr
import numpy as np
import dask.array as da
import uuid
import h5py
from ..core.utilities import *
from ..core.progressbar import *
from .missing_values import *


def regrid(X, lons, lats, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', use_dask=False, feat_chunks=1, samp_chunks=1 ):
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	X1 = fill_space_mean(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim )
	X1 = X1.chunk({x_feature_dim: max(X.shape[list(X.dims).index(x_feature_dim)] // feat_chunks, 1), x_sample_dim: max(X.shape[list(X.dims).index(x_sample_dim)] // samp_chunks, 1) }).transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)

	if use_dask:
		id = str(uuid.uuid4())
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
	return xr.DataArray(data=results, coords=coords, dims=X1.dims)


def regrid_chunk(X, lats, lons, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', use_dask=False, hdf=None, feature_ndx=0, sample_ndx=0 ):
	res = []
	data = X.values
	for i in range(data.shape[0]):
		res.append([])
		for j in range(data.shape[1]):
			interp_func = interp2d(X.coords[x_lon_dim].values, X.coords[x_lat_dim].values, data[i,j, :,:], kind='linear')
			res[i].append(interp_func(lons, lats))
	res = np.asarray(res)
	if use_dask:
		hdf.create_dataset('data_{}_{}'.format(feature_ndx, sample_ndx), data=res)
		return 'data_{}_{}'.format(feature_ndx, sample_ndx)
	else:
		return res








def gaussian_smooth(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, feature_chunks=1, sample_chunks=1 ):
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	#X1 = fill_space_mean(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim )
	X1 = X.chunk({x_feature_dim: max(X.shape[list(X.dims).index(x_feature_dim)] // feature_chunks, 1), x_sample_dim: max(X.shape[list(X.dims).index(x_sample_dim)] // sample_chunks, 1) }).transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)

	if use_dask:
		id = str(uuid.uuid4())
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
			results[i].append(gaussian_smooth_chunk(X1.isel(**x_isel), x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , use_dask=use_dask, kernel=kernel, hdf=hdf))
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


def gaussian_smooth_chunk(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, hdf=None ):
	res = []
	data = X.values
	for i in range(data.shape[0]):
		res.append([])
		for j in range(data.shape[1]):
			res[i].append(cv2.GaussianBlur(data[i, j, :, :], kernel,0))
	res = np.asarray(res)
	if use_dask:
		hdf.create_dataset('data_{}_{}'.format(feature_ndx, sample_ndx), data=res)
		return 'data_{}_{}'.format(feature_ndx, sample_ndx)
	else:
		return res
