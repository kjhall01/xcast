import datetime as dt
import xarray as xr
from ..preprocessing import *
from ..core import *
from ..downscaling import *
import dask.array as da
import dask as d
from dask_ml.wrappers import Incremental
import uuid
from dask import delayed, compute


class PointWiseMME:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.scaler_x = None
		self.scaler_y = None
		self.pca_x = None
		self.pointwise = None

	def fit(self, X, Y, mme_func, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False,  pca_x=False,  rescale_x=None, rescale_y=None, fill='mean', missing_value=-1, t_chunks=20, x_chunks=10, y_chunks=10):
		assert self.pointwise is None, '{} Cannot Re-Fit a PointWiseMME Type'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		assert 'X' in y_coords.keys() and 'Y' in y_coords.keys() and 'T' in y_coords.keys(), 'XYT must be indicated in y_coords'
		self.mme_func = mme_func.__name__
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M']).chunk({x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)), x_coords['M']: len(X.coords[x_coords['M']].values), x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks)), x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks))  })

		if 'M' not in y_coords.keys():
			y_coords['M'] = 'M'
		Y = standardize_dims(Y, y_coords, verbose=verbose)
		Y = Y.transpose(y_coords['Y'], y_coords['X'], y_coords['T'], y_coords['M']).chunk({y_coords['T']: max(1, int(len(Y.coords[y_coords['T']].values) / t_chunks)), y_coords['M']: len(Y.coords[y_coords['M']].values), y_coords['Y']: max(1, int(len(Y.coords[y_coords['Y']].values) / y_chunks)), y_coords['X']: max(1, int(len(Y.coords[y_coords['X']].values) / x_chunks))  })

		count = 0
		if verbose > 1:
			total = len(X.chunks[x_coords['Y']]) * len(X.chunks[x_coords['X']])
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')


		self.pointwise = []
		for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
			self.pointwise.append([])
			indx = 0
			for ii in range(i):
				indx += X.chunks[x_coords['Y']][ii]
			for j in range(len(X.chunks[x_coords['X']])): # chunks in X
				#print('{} {}'.format(i, j), end='\r')
				jndx = 0
				for jj in range(j):
					jndx += X.chunks[x_coords['X']][jj]
				count += 1
				if verbose > 1:
					print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				self.pointwise[i].append(PointWiseMMEOne(**self.kwargs))
				iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
				iseldicty = {y_coords['X']: slice(jndx, jndx+Y.chunks[y_coords['X']][j]), y_coords['Y']: slice(indx, indx+Y.chunks[y_coords['Y']][i])  }
				self.pointwise[i][j].fit(X.isel(**iseldict), Y.isel(**iseldicty), mme_func, x_coords=x_coords, y_coords=y_coords,  verbose=verbose-1)
		if verbose > 1:
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))


	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False,  pca_x=False,  rescale_x=None, rescale_y=None, fill='mean', missing_value=-1, t_chunks=20, x_chunks=10, y_chunks=10):
		assert self.pointwise is not None, '{} Must Fit a PointWiseMME Type before predicting'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		tf_uuid = str(uuid.uuid4())
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M']).chunk({x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)), x_coords['M']: len(X.coords[x_coords['M']].values), x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks)), x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks))  })

		count = 0
		if verbose > 1:
			total = len(X.chunks[x_coords['Y']]) * len(X.chunks[x_coords['X']])
			print('{} PointWiseMME Predict for {}: ['.format(dt.datetime.now(), self.mme_func) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		ids = []
		for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
			indx = 0
			for ii in range(i):
				indx += X.chunks[x_coords['Y']][ii]
			for j in range(len(X.chunks[x_coords['X']])): # chunks in X
				jndx = 0
				for jj in range(j):
					jndx += X.chunks[x_coords['X']][jj]
				count += 1
				if verbose > 1:
					print('{} PointWiseMME Predict for {}: ['.format(dt.datetime.now(), self.mme_func) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
				self.pointwise[i][j].predict(X.isel(**iseldict), x_coords=x_coords,  verbose=verbose-1).to_netcdf(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
				ids.append(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))

		if verbose > 1:
			print('{} PointWiseMME Predict for {}'.format(dt.datetime.now(), self.mme_func))
		return xr.open_mfdataset(ids, chunks='auto', parallel=True)

class PointWiseMMEOne:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.scaler_x = None
		self.scaler_y = None
		self.pca_x = None
		self.models = None


	def fit(self, X, Y, mme_func, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False,  pca_x=False,  rescale_x=None, rescale_y=None, fill='mean', missing_value=-1, t_chunks=20, x_chunks=10, y_chunks=10):
		assert self.models is None, '{} Cannot Re-Fit a PointWiseMME Type'.format(dt.datetime.now())
		X = standardize_dims(X, x_coords, verbose=verbose)
		Y = standardize_dims(Y, y_coords, verbose=verbose)


		self.mme_func = mme_func
		count=0
		if verbose > 2:
			total = len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		if str(rescale_y).upper() == 'NORMAL':
			self.scaler_y = NormalScaler(**self.kwargs)
			self.scaler_y.fit(Y, y_coords, verbose=verbose)
			Y = self.scaler_y.transform(Y, y_coords, verbose=verbose)
		elif str(rescale_y).upper() == 'MINMAX':
			self.scaler_y = MinMaxScaler(**self.kwargs)
			self.scaler_y.fit(Y, y_coords, verbose=verbose)
			Y = self.scaler_y.transform(Y, y_coords, verbose=verbose)
		else:
			pass

		if str(rescale_x).upper() == 'NORMAL':
			self.scaler_x = NormalScaler(**self.kwargs)
			self.scaler_x.fit(X, x_coords, verbose=verbose)
			X = self.scaler_x.transform(X, x_coords, verbose=verbose)
		elif str(rescale_x).upper() == 'MINMAX':
			self.scaler_x = MinMaxScaler(**kwargs)
			self.scaler_x.fit(X, x_coords, verbose=verbose)
			X = self.scaler_x.transform(X, x_coords, verbose=verbose)
		else:
			pass

		if pca_x:
			assert 'n_components' in self.kwargs.keys(), '{} PrincipalComponents requires n_components keyword'.format(dt.datetime.now())
			self.pca_x = PrincipalComponents(**self.kwargs)
			self.pca_x.fit(X, x_coords, verbose=verbose)
			X = self.pca_x.transform(X, x_coords, verbose=verbose)

		xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]
		self.shape = {}
		for coord in x_coords.keys():
			if coord in X.coords:
				self.shape[coord] = len(X.coords[x_coords[coord]].values)

		self.kwargs['x_train_shape'] = self.shape['M'] if 'M' in self.shape.keys() else 1
		self.kwargs['y_train_shape'] = 1 # were assuming 1 predictand

		x_data = getattr(X, xvarname).values
		y_data = getattr(Y, yvarname).values
		models = []
		for i in range(x_data.shape[0]):
			models.append([])
			for j in range(x_data.shape[1]):
				count += 1
				if count % 55 == 0 and verbose > 2:
					print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				models[i].append(mme_func(**self.kwargs))
				#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i}
				#isel_y_dict = {y_coords['X']: j, x_coords['Y']: i}
				x_train = x_data[i, j, :, :] #getattr(X, xvarname).isel(**isel_x_dict).values
				y_train = y_data[i, j, :, :] #getattr(Y, yvarname).isel(**isel_y_dict).values.T
				#print(x_train.shape, y_train.shape)
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				if len(y_train.shape) < 2:
					y_train = y_train.reshape(-1,1)
				if x_train.shape[1] > x_train.shape[0]:
					x_train = x_train.T
				if y_train.shape[1] > y_train.shape[0]:
					y_train = y_train.T
				models[i][j].fit(x_train, y_train)
		if verbose > 2:
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
		self.models = models

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean', missing_value=-1,t_chunks=20, x_chunks=10, y_chunks=10):
		assert self.models is not None, '{} Must Fit PointWiseMME Type before Predicting'.format(dt.datetime.now())
		X = standardize_dims(X, x_coords, verbose=verbose)
		#if 'M' in x_coords.keys():
		#	X = X.chunk({x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)), x_coords['M']: len(X.coords[x_coords['M']].values), x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks)), x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks))  })
		#else:
		#	X = X.chunk({x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)),  x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks)), x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks)) })
		xvarname = [i for i in X.data_vars][0]
		x_data = getattr(X, xvarname).values
		#y_data = getattr(Y, yvarname).values
		#x_coords_slice = {'X':x_coords['X'], 'Y':x_coords['Y'], 'T':x_coords['T']}
		tf_uuid = str(uuid.uuid4())
		count=0
		if verbose > 2:
			total = len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
			print('{} Predicting PointWiseMME for {}: ['.format(dt.datetime.now(), self.mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		x_shape = {}
		for coord in x_coords.keys():
			x_shape[coord] = len(X.coords[x_coords[coord]].values)

		for dim in self.shape.keys():
			if dim != 'T':
				assert x_shape[dim] == self.shape[dim], '{} Mismatched {}-Dim: {} on X, but {} on Train'.format(dt.datetime.now(), dim, x_shape[dim], self.shape[dim])

		if self.scaler_x is not None:
			X = self.scaler_x.transform(X, x_coords, verbose=verbose)

		if self.pca_x is not None:
			X = self.pca_x.transform(X, x_coords, verbose=verbose)

		ret, ids = [], []
		for i in range(x_data.shape[0]):
			ret.append([])
			for j in range(x_data.shape[1]):
				count += 1
				if count % 55 == 0 and verbose > 2:
					print('{} Predicting PointWiseMME for {}: ['.format(dt.datetime.now(), self.mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i}
				x_train = x_data[i, j, :, :]#getattr(X, xvarname).isel(**isel_x_dict).values.T
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				if x_train.shape[1] > x_train.shape[0]:
					x_train = x_train.T

				nnn = self.models[i][j].predict(x_train)
				retval = np.squeeze(nnn)
				ret[i].append(retval)
			coords = {x_coords['X']:X.coords[x_coords['X']].values, x_coords['Y']:[X.coords[x_coords['Y']].values[i]], x_coords['T']:X.coords[x_coords['T']].values}
			data_vars = {xvarname: ([x_coords['Y'], x_coords['X'], x_coords['T']], np.asarray([ret[i]]))}
			xr.Dataset(data_vars, coords=coords).to_netcdf(Path().home() / '.xcast_cache' / '{}_{}.nc'.format(tf_uuid, i))
			ids.append(Path().home() / '.xcast_cache' / '{}_{}.nc'.format(tf_uuid, i))
		if verbose > 2:
			print('{} Predicting PointWiseMME for {}: ['.format(dt.datetime.now(), self.mme_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))

		ret =  xr.open_mfdataset(ids, chunks='auto', parallel=True)

		if self.scaler_y is not None:
			ret = self.scaler_y.inverse_transform(ret, x_coords)
		return ret
