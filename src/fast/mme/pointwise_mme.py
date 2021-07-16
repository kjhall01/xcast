import datetime as dt
import xarray as xr
from ..preprocessing import *
from ..core import *
from ..downscaling import *
import dask.array as da


class ProbabilisticPointWiseMME:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.scaler_x = None
		self.scaler_y = None
		self.pca_x = None
		self.models = None

	def fit(self, X, Y, mme_func, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False,  pca_x=False,  rescale_x=None, rescale_y=None, fill='mean', missing_value=-1, one_hot=False):
		assert self.models is None, '{} Cannot Re-Fit a PointWiseMME Type'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		assert 'X' in y_coords.keys() and 'Y' in y_coords.keys() and 'T' in y_coords.keys(), 'XYT must be indicated in y_coords'
		self.mme_func = mme_func
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])

		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		Y = standardize_dims(Y, y_coords, verbose=verbose)
		self.scaler_y = ProbabilisticScaler(**self.kwargs)
		self.scaler_y.fit(Y, y_coords, verbose=verbose)
		Y = self.scaler_y.transform(Y, y_coords, verbose=verbose)
		y_coords['C'] = 'C'
		if one_hot:
			Y = Y.mean(y_coords['M']).transpose(y_coords['Y'], y_coords['X'], y_coords['T'], y_coords['C'])
		else:
			Y = Y.mean(y_coords['M']).transpose(y_coords['Y'], y_coords['X'], y_coords['T'], y_coords['C']).argmax(y_coords['C'])


		count=0
		if verbose > 1:
			total = len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')




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
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])
		#Y = Y.mean(y_coords['M']).transpose(y_coords['Y'], y_coords['X'], y_coords['T'], y_coords['C'])
		x_data = getattr(X, xvarname).values
		y_data = getattr(Y, yvarname).values
		self.kwargs['y_train_shape'] = y_data.shape[-1] # were assuming 1 predictand

		models = []
		for i in range(len(X.coords[x_coords['Y']].values)):
			models.append([])
			for j in range(len(X.coords[x_coords['X']].values)):
				count += 1
				if  verbose > 1:
					print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				models[i].append(mme_func(**self.kwargs))
				x_train = x_data[i, j, :, :] #getattr(X, xvarname).isel(**isel_x_dict).values
				y_train = y_data[i, j, :] #getattr(Y, yvarname).isel(**isel_y_dict).values.T
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
		if verbose > 1:
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
		self.models = models

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean', missing_value=-1):
		assert self.models is not None, '{} Must Fit PointWiseMME Type before Predicting'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])


		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		x_coords_slice = {'X':x_coords['X'], 'Y':x_coords['Y'], 'T':x_coords['T']}

		count=0
		if verbose > 1:
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
		#X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])
		xvarname = [i for i in X.data_vars][0]
		x_data = getattr(X, xvarname).values
		ret = []
		for i in range(len(X.coords[x_coords['Y']].values)):
			ret.append([])
			for j in range(len(X.coords[x_coords['X']].values)):
				count += 1
				if  verbose > 1:
					print('{} Predicting PointWiseMME for {}: ['.format(dt.datetime.now(), self.mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				x_train = x_data[i, j, :, :]#getattr(X, xvarname).isel(**isel_x_dict).values.T
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				if x_train.shape[1] > x_train.shape[0]:
					x_train = x_train.T

				retval = self.models[i][j].predict_proba(x_train)
				assert len(retval[0]) == 3, 'Not all classes present in set '
				ret[i].append(retval)
		if verbose > 1:
			print('{} Predicting PointWiseMME for {}: ['.format(dt.datetime.now(), self.mme_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
		coords = {x_coords['X']:X.coords[x_coords['X']].values, x_coords['Y']:X.coords[x_coords['Y']].values, x_coords['T']:X.coords[x_coords['T']].values, 'C': [0, 1, 2]}
		data_vars = {xvarname: ([x_coords['Y'], x_coords['X'],  x_coords['T'], 'C'], np.asarray(ret))}
		ret1 =  xr.Dataset(data_vars, coords=coords)
		return ret1

class DeterministicPointWiseMME:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.scaler_x = None
		self.scaler_y = None
		self.pca_x = None
		self.models = None


	def fit(self, X, Y, mme_func, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False,  pca_x=False,  rescale_x=None, rescale_y=None, fill='mean', missing_value=-1):
		assert self.models is None, '{} Cannot Re-Fit a PointWiseMME Type'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		assert 'X' in y_coords.keys() and 'Y' in y_coords.keys() and 'T' in y_coords.keys(), 'XYT must be indicated in y_coords'
		self.mme_func = mme_func
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])

		if 'M' not in y_coords.keys():
			y_coords['M'] = 'M'
		Y = standardize_dims(Y, y_coords, verbose=verbose)
		Y = Y.transpose(y_coords['Y'], y_coords['X'], y_coords['T'], y_coords['M'])

		count=0
		if verbose > 1:
			total = len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		if str(rescale_y).upper() == 'NORMAL':
			self.scaler_y = NormalScaler(**self.kwargs)
			self.scaler_y.fit(Y, y_coords, verbose=verbose)
			Y = self.scaler_y.transform(Y, y_coords, verbose=verbose)
		elif str(rescale_y).upper() == 'MINMAX':
			self.scaler_y = MinMaxScaler(**self.kwargs)
			print(Y)
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
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])
		Y = Y.transpose(y_coords['Y'], y_coords['X'], y_coords['T'], y_coords['M'])
		x_data = getattr(X, xvarname).values
		y_data = getattr(Y, yvarname).values
		models = []
		for i in range(len(X.coords[x_coords['Y']].values)):
			models.append([])
			for j in range(len(X.coords[x_coords['X']].values)):
				count += 1
				if  verbose > 1:
					print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				models[i].append(mme_func(**self.kwargs))
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
		if verbose > 1:
			print('{} Fitting PointWiseMME for {}: ['.format(dt.datetime.now(), mme_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
		self.models = models

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean', missing_value=-1):
		assert self.models is not None, '{} Must Fit PointWiseMME Type before Predicting'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])


		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		x_coords_slice = {'X':x_coords['X'], 'Y':x_coords['Y'], 'T':x_coords['T']}

		count=0
		if verbose > 1:
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
		#X = X.transpose(x_coords['Y'], x_coords['X'], x_coords['T'], x_coords['M'])
		xvarname = [i for i in X.data_vars][0]
		x_data = getattr(X, xvarname).values
		ret = []
		for i in range(len(X.coords[x_coords['Y']].values)):
			ret.append([])
			for j in range(len(X.coords[x_coords['X']].values)):
				count += 1
				if  verbose > 1:
					print('{} Predicting PointWiseMME for {}: ['.format(dt.datetime.now(), self.mme_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

				x_train = x_data[i, j, :, :]#getattr(X, xvarname).isel(**isel_x_dict).values.T
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				if x_train.shape[1] > x_train.shape[0]:
					x_train = x_train.T

				nnn = self.models[i][j].predict(x_train)
				retval = np.squeeze(nnn)
				ret[i].append(retval)
		if verbose > 1:
			print('{} Predicting PointWiseMME for {}: ['.format(dt.datetime.now(), self.mme_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
		coords = {x_coords['X']:X.coords[x_coords['X']].values, x_coords['Y']:X.coords[x_coords['Y']].values, x_coords['T']:X.coords[x_coords['T']].values}
		data_vars = {xvarname: ([x_coords['Y'], x_coords['X'], x_coords['T']], np.asarray(ret))}
		ret1 =  xr.Dataset(data_vars, coords=coords)
		x_coords_slice = { 'X': x_coords['X'], 'Y':x_coords['Y'], 'T':x_coords['T']}
		if self.scaler_y is not None:
			ret1 = self.scaler_y.inverse_transform(ret1, x_coords)
		return ret1
