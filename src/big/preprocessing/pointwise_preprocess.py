import datetime as dt
import xarray as xr
from ..core import *

class PointWisePreprocess:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.models = None

	def fit(self, X,  preprocess_func, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, is_pca=False):
		assert self.models is None, '{} Cannot Re-Fit a PointWisePP Type'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		self.preprocess_func = preprocess_func
		X = standardize_dims(X, x_coords, verbose=verbose)
		xvarname = [i for i in X.data_vars][0]
		self.shape = {}
		for coord in x_coords.keys():
			self.shape[coord] = len(X.coords[x_coords[coord]].values)

		x_data = getattr(X, xvarname).transpose(x_coords['M'],  x_coords['Y'], x_coords['X'], x_coords['T']).values
		if 'M' in x_coords.keys() and not is_pca:
			count=0
			if verbose > 3:
				total = len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
				print('{} Fitting PointWisePreProcess for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')


			models = []
			for k in range(x_data.shape[0]):
				models.append([])
				for i in range(x_data.shape[1]):
					models[k].append([])
					for j in range(x_data.shape[2]):
						models[k][i].append(self.preprocess_func(**self.kwargs))
						#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i, x_coords['M']: k}
						x_train = x_data[k, i, j, :]
						if len(x_train.shape) < 2:
							x_train = x_train.reshape(-1,1)
						if x_train.shape[1] > x_train.shape[0]:
							x_train = x_train.T
						models[k][i][j].fit(x_train)
						count += 1
						if verbose > 3:
							print('{} Fitting PointWisePreProcess for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
			if verbose > 3:
				print('{} Fitting PointWisePreProcess for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
			self.models = models
		else:
			count=0
			if verbose > 3:
				total =  len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
				print('{} Fitting PointWisePreProcess for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

			models = []
			for i in range(len(X.coords[x_coords['Y']].values)):
				models.append([])
				for j in range(len(X.coords[x_coords['X']].values)):
					models[i].append(self.preprocess_func(**self.kwargs))
					#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i}
					#x_train = getattr(X, xvarname).isel(**isel_x_dict).values.T
					x_train = x_data[i, j, :]
					if len(x_train.shape) < 2:
						x_train = x_train.reshape(-1,1)
					if x_train.shape[1] > x_train.shape[0]:
						x_train = x_train.T
					models[i][j].fit(x_train)
					count += 1
					if  verbose > 3:
						print('{} Fitting PointWisePreProcess for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
			if verbose > 3:
				print('{} Fitting PointWisePreProcess for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
			self.models = models

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},   verbose=False, is_pca=False):
		assert self.models is not None, '{} Must Fit PointWiseMME Type before Predicting'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		x_shape = {}
		for coord in x_coords.keys():
			x_shape[coord] = len(X.coords[x_coords[coord]].values)
		for dim in x_coords.keys():
			if dim != 'T':
				assert x_shape[dim] == self.shape[dim], '{} Mismatched {}-Dim: {} on X, but {} on Train'.format(dt.datetime.now(), dim, x_shape[dim], self.shape[dim])
		xvarname = [i for i in X.data_vars][0]
		for coord in x_coords.keys():
			self.shape[coord] = len(X.coords[x_coords[coord]].values)
		x_data = getattr(X, xvarname).transpose(x_coords['M'],  x_coords['Y'], x_coords['X'], x_coords['T']).values

		if 'M' in x_coords.keys() and not is_pca:
			count=0
			if verbose > 3:
				total = len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
				print('{} PointWisePreProcess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

			ret = []
			for k in range(len(X.coords[x_coords['M']].values)):
				ret.append([])
				for i in range(len(X.coords[x_coords['Y']].values)):
					ret[k].append([])
					for j in range(len(X.coords[x_coords['X']].values)):
						#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i, x_coords['M']:k}
						x_train = x_data[k, i, j, :]#getattr(X, xvarname).isel(**isel_x_dict).values.T
						retval = self.models[k][i][j].transform(x_train)
						ret[k][i].append(np.squeeze(retval))
						count += 1
						if  verbose > 3:
							print('{} PointWisePreProcess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
			if verbose > 3:
				print('{} PointWisePreprocess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
			coords = {x_coords['X']:X.coords[x_coords['X']].values, x_coords['Y']:X.coords[x_coords['Y']].values, x_coords['T']:X.coords[x_coords['T']].values, x_coords['M']:X.coords[x_coords['M']].values}
			data_vars = {(xvarname + '_transformed'): ([x_coords['M'], x_coords['Y'], x_coords['X'], x_coords['T']], np.asarray(ret))}
			return xr.Dataset(data_vars, coords=coords)
		else:
			count=0
			if verbose > 3:
				total =  len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
				print('{} PointWisePreprocess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

			ret = []
			for i in range(len(X.coords[x_coords['Y']].values)):
				ret.append([])
				for j in range(len(X.coords[x_coords['X']].values)):
					#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i}
					x_train = x_data[i, j, :] #getattr(X, xvarname).isel(**isel_x_dict).values.T
					retval = self.models[i][j].transform(x_train)
					ret[i].append(np.squeeze(retval))
					count += 1
					if  verbose > 3:
						print('{} PointWisePreProcess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
			if verbose > 3:
				print('{} PointWisePreProcess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
			coords = {x_coords['X']:X.coords[x_coords['X']].values, x_coords['Y']:X.coords[x_coords['Y']].values, x_coords['T']:X.coords[x_coords['T']].values}
			data_vars = {(xvarname + '_transformed'): ([x_coords['Y'], x_coords['X'], x_coords['T']], np.asarray(ret))}
			return xr.Dataset(data_vars, coords=coords)


	def inverse_transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False, is_pca=False):
		assert self.models is not None, '{} Must Fit PointWiseMME Type before Predicting'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys(), 'XYT must be indicated in x_coords'
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'

		X = standardize_dims(X, x_coords, verbose=verbose)
		x_shape = {}
		for coord in x_coords.keys():
			x_shape[coord] = len(X.coords[x_coords[coord]].values)
		for dim in x_coords.keys():
			if dim != 'T':
				assert x_shape[dim] == self.shape[dim], '{} Mismatched {}-Dim: {} on X, but {} on Train'.format(dt.datetime.now(), x_shape[dim], self.shape[dim])
		xvarname = [i for i in X.data_vars][0]
		for coord in x_coords.keys():
			self.shape[coord] = len(X.coords[x_coords[coord]].values)
		x_data = getattr(X, xvarname).transpose(x_coords['M'],  x_coords['Y'], x_coords['X'], x_coords['T']).values

		if 'M' in x_coords.keys() and not is_pca:
			count=0
			if verbose > 3:
				total = len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
				print('{} PointWisePreProcess Inverse Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

			ret = []
			for k in range(len(X.coords[x_coords['M']].values)):
				ret.append([])
				for i in range(len(X.coords[x_coords['Y']].values)):
					ret[k].append([])
					for j in range(len(X.coords[x_coords['X']].values)):
						#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i, x_coords['M']:k}
						x_train = x_data[k, i, j, :]#getattr(X, xvarname).isel(**isel_x_dict).values.T
						retval = self.models[k][i][j].inverse_transform(x_train)
						ret[k][i].append(np.squeeze(retval))
						count += 1
						if  verbose > 3:
							print('{} PointWisePreProcess Inverse Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
			if verbose > 3:
				print('{} PointWisePreProcess Inverse Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
			coords = {x_coords['X']:X.coords[x_coords['X']].values, x_coords['Y']:X.coords[x_coords['Y']].values, x_coords['T']:X.coords[x_coords['T']].values, x_coords['M']:X.coords[x_coords['M']].values}
			data_vars = {(xvarname + '_recovered'): ([x_coords['M'], x_coords['Y'], x_coords['X'], x_coords['T']], np.asarray(ret))}
			return xr.Dataset(data_vars, coords=coords)
		else:
			count=0
			if verbose > 3:
				total = len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
				print('{} PointWisePreProcess Inverse Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

			ret = []
			for i in range(len(X.coords[x_coords['Y']].values)):
				ret.append([])
				for j in range(len(X.coords[x_coords['X']].values)):
					#isel_x_dict = {x_coords['X']: j, x_coords['Y']: i}
					x_train = x_data[i, j, :] #getattr(X, xvarname).isel(**isel_x_dict).values.T
					retval = self.models[i][j].inverse_transform(x_train)
					ret[i].append(np.squeeze(retval))
					count += 1
					if verbose > 3:
						print('{} PointWisePreProcess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
			if verbose > 3:
				print('{} PointWisePreProcess Transform for {}: ['.format(dt.datetime.now(), self.preprocess_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
			coords = {x_coords['X']:X.coords[x_coords['X']].values, x_coords['Y']:X.coords[x_coords['Y']].values, x_coords['T']:X.coords[x_coords['T']].values}
			data_vars = {(xvarname + '_recovered'): ([x_coords['Y'], x_coords['X'], x_coords['T']], np.asarray(ret))}
			return xr.Dataset(data_vars, coords=coords)
