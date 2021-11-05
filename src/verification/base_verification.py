from sklearn.metrics import mean_squared_error
import xarray as xr
import numpy as np
import dask.array as da
import uuid
import h5py
from ..core.utilities import *
from ..core.progressbar import *

import dask.diagnostics as dd

def dask_expand_dims(x, axis=-1):
	new_shape = [i for i in x.shape]
	new_shape.insert( axis, 1)
	return da.from_array([x]).reshape(*new_shape)

def reformat_vector(x, fmt):
	assert fmt in ['flat', 'row', 'col'], 'invalid vector format: {} - must be in ["flat", "row", "col"]'.format(fmt)
	if len(x.shape) == 2 and fmt == 'col':
		return x
	x = np.asarray(x)
	dims, found = len(x.shape), 0
	for i in x.shape:
		if found:
			assert i == 1, 'x can only have one dimension of size > 1'
		else:
			if i != 1:
				found = 1
	x = np.squeeze(x)
	if len(x.shape) == 0:
		if fmt == 'flat':
			return np.expand_dims(x, 0).astype(float)
		elif fmt == 'row':
			return np.expand_dims(np.expand_dims(x, 0), 0).astype(float)
		else: #col
			return np.expand_dims(np.expand_dims(x, 0), 0).T.astype(float)
	elif len(x.shape) == 1:
		if fmt == 'flat':
			return x.astype(float)
		elif fmt == 'row':
			return np.expand_dims(x, 0).astype(float)
		else: #col
			return np.expand_dims(x, 0).T.astype(float)
	else:
		assert False, 'np.squeeze(x).shape must be of length 0, or 1. (no more than 1 dimension with size > 1)'

def apply_func_to_block(x_data, y_data, func1=mean_squared_error, kwargs={}, n=1, xfmt='col', yfmt='col', opfmt='flat'):
	assert len(x_data.shape) == 4, 'x_data must be 4D'
	assert len(y_data.shape) == 4, 'y_data must be 4D'
	ret = []
	for i in range(x_data.shape[0]):
		ret.append([])
		for j in range(x_data.shape[1]):
			x_train = reformat_vector(x_data[i, j, :, :], xfmt)
			y_train = reformat_vector(y_data[i, j, :, :], yfmt)
			x = np.squeeze(func1(x_train, y_train, **kwargs))
			ret[i].append(x)
		#ret[i] = np.asarray(ret[i]).astype(float)
	ret = np.asarray(ret)
	if len(ret.shape) < 3:
		ret = np.expand_dims(ret, axis=2)
	if len(ret.shape) < 4:
		ret = np.expand_dims(ret, axis=3)
	return ret

def metric(func):
	def func1(X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False, rechunk=False, parallel_in_memory=True, n=1, xfmt='col', yfmt='col', **kwargs):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		assert X.shape[list(X.dims).index(x_lat_dim)] == Y.shape[list(Y.dims).index(y_lat_dim)], 'Xcast metrics requires X to have the same latitudinal resolution as Y'
		assert X.shape[list(X.dims).index(x_lon_dim)] == Y.shape[list(Y.dims).index(y_lon_dim)], 'Xcast metrics requires X to have the same longitudinal resolution as Y'
		assert X.shape[list(X.dims).index(x_sample_dim)] == Y.shape[list(Y.dims).index(y_sample_dim)], 'Xcast metrics requires X to have the same temporal resolution as Y'
		#assert X.shape[list(X.dims).index(x_feature_dim)] == Y.shape[list(Y.dims).index(y_feature_dim)] and Y.shape[list(Y.dims).index(y_feature_dim)] == 1, 'Xcast metrics requires X and Y to have a feature dimension of size 1'
		X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		if rechunk:
			X1 = X1.chunk({x_lat_dim: max(X1.shape[list(X1.dims).index(x_lat_dim)] // lat_chunks,1), x_lon_dim: max(X1.shape[list(X1.dims).index(x_lon_dim)] // lon_chunks,1), x_feature_dim: -1, x_sample_dim: -1})
			Y1 = Y1.chunk({y_lat_dim: max(Y1.shape[list(Y1.dims).index(y_lat_dim)] // lat_chunks, 1), y_lon_dim: max(Y1.shape[list(Y1.dims).index(y_lon_dim)] // lon_chunks,1), y_feature_dim: -1, y_sample_dim: -1})
			
		x_data, y_data = X1.data, Y1.data
		if verbose:
			with dd.ProgressBar():
				scores = da.map_blocks(apply_func_to_block, x_data, y_data, drop_axis=[2,3], new_axis=[3,4], func1=func, kwargs=kwargs,n=n, xfmt=xfmt, yfmt=yfmt, meta=np.array((), dtype=float) ).compute()
		else:
			scores = da.map_blocks(apply_func_to_block, x_data, y_data, drop_axis=[2,3], new_axis=[3,4], func1=func,  kwargs=kwargs,n=n, xfmt=xfmt, yfmt=yfmt,  meta=np.array((), dtype=float)).compute()
		return xr.DataArray(data=scores, dims=[x_lat_dim, x_lon_dim , x_feature_dim, 'SKILLDIM'], coords={x_lat_dim: X1.coords[x_lat_dim].values, x_lon_dim: X1.coords[x_lon_dim].values, x_feature_dim: [i for i in range(scores.shape[2])], 'SKILLDIM': [i for i in range(scores.shape[3])] } , attrs=X1.attrs, name = func.__name__)
	func1.__name__ = func.__name__
	return func1
