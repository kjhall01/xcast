from .pointwise_preprocess import PointWisePreprocess
from ..core import *
import numpy as np
import datetime as dt
from sklearn.decomposition import PCA

class PrincipalComponents:
	def __init__(self, n_components=0.9, **kwargs):
		self.pct = max(int(n_components), 1)
		self.pointwise = PointWisePreprocess(n_components=self.pct)

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		self.pointwise.fit(X, PCA, x_coords=x_coords, verbose=verbose, is_pca=True)

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		return self.pointwise.transform(X, x_coords=x_coords,  verbose=verbose, is_pca=True)

	def inverse_transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		return self.pointwise.inverse_transform(X, x_coords=x_coords,  verbose=verbose, is_pca=True)

class SpatialPrincipalComponents:
	def __init__(self, n_components=5):
		assert n_components is None or type(n_components) is int, '{} invalid n_components arg, must be None or integer number of components'.format(dt.datetime.now())
		self.n_components = n_components
		self.pcas = []

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		X = standardize_dims(X, x_coords)
		varname = [i for i in X.data_vars][0]
		self.x_size, self.y_size = len(X.coords[x_coords['X']].values), len(X.coords[x_coords['Y']].values)
		if 'M' in x_coords.keys():
			for i in range(len(X.coords[x_coords['M']].values)):
				iseldict = {x_coords['M']: i}
				values = []
				for j in range(len(X.coords[x_coords['T']].values)):
					iseldict[x_coords['T']] = j
					vals = np.squeeze(getattr(X, varname).isel(**iseldict).transpose(x_coords['X'], x_coords['Y']).values.reshape(self.x_size*self.y_size))
					values.append(vals)
				values = np.asarray(values)
				self.pcas.append(PCA(n_components=self.n_components))
				self.pcas[i].fit(values)

		else:
			iseldict = {}
			values = []
			for j in range(len(X.coords[x_coords['T']].values)):
				iseldict[x_coords['T']] = j
				vals = np.squeeze(getattr(X, varname).isel(**iseldict).transpose(x_coords['X'], x_coords['Y']).values.reshape(self.x_size*self.y_size))
				values.append(vals)
			self.pcas.append(PCA(n_components=self.n_components))
			self.pcas[0].fit(values)

		eofs = []
		for i in range(len(self.pcas)):
			eofs_i = self.pcas[i].components_.reshape(self.pcas[i].n_components_, self.y_size*self.x_size )
			eofs_i = np.asarray([eofs_i[x].reshape(self.x_size, self.y_size).T for x in range(eofs_i.shape[0])])
			eofs.append(eofs_i)
		eofs = np.asarray(eofs)
		coords = {
			'MODE': [k for k in range(self.pcas[i].n_components_)],
			x_coords['X']: X.coords[x_coords['X']].values,
			x_coords['Y']: X.coords[x_coords['Y']].values
		}
		if 'M' in x_coords.keys():
			coords[x_coords['M']] =  X.coords[x_coords['M']].values
		else:
			coords['M'] = [0]
			x_coords['M'] = 'M'

		data_vars = {'EOFS': ([x_coords['M'], 'MODE', x_coords['Y'], x_coords['X']], eofs)}
		self.eofs = xr.Dataset(data_vars, coords=coords).EOFS

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		X = standardize_dims(X, x_coords)
		varname = [i for i in X.data_vars][0]
		assert len(X.coords[x_coords['X']].values) == self.x_size, '{} Wrong Size Spatial Data - should be {} x {}'.format(dt.datetime.now(), self.x_size, self.y_size)
		assert len(X.coords[x_coords['Y']].values) == self.y_size, '{} Wrong Size Spatial Data - should be {} x {}'.format(dt.datetime.now(), self.x_size, self.y_size)
		if 'M' in x_coords.keys():
			ret = []
			for i in range(len(X.coords[x_coords['M']].values)):
				iseldict = {x_coords['M']: i}
				values = []
				for j in range(len(X.coords[x_coords['T']].values)):
					iseldict[x_coords['T']] = j
					vals = np.squeeze(getattr(X, varname).isel(**iseldict).transpose(x_coords['X'], x_coords['Y']).values.reshape(self.x_size*self.y_size))
					values.append(vals)

				pc = self.pcas[i].transform(values)[:, :self.n_components]
				ret.append(pc)
			ret = np.asarray(ret)
			reshaped = []
			for i in range(self.y_size):
				reshaped.append([])
				for j in range(self.x_size):
					reshaped[i].append(ret)
			reshaped = np.asarray(reshaped)
			coords = {
				'MODE': [k for k in range(self.n_components)],
				x_coords['X']: X.coords[x_coords['X']].values,
				x_coords['Y']: X.coords[x_coords['Y']].values,
				x_coords['T']: X.coords[x_coords['T']].values,
				x_coords['M']: X.coords[x_coords['M']].values,
			}
			data_vars = {'PCS': ([x_coords['Y'], x_coords['X'], x_coords['M'], x_coords['T'], 'MODE'], reshaped )}
			return xr.Dataset(data_vars, coords=coords)
		else:
			iseldict = {}
			values = []
			for j in range(len(X.coords[x_coords['T']].values)):
				iseldict[x_coords['T']] = j
				vals = np.squeeze(getattr(X, varname).isel(**iseldict).transpose(x_coords['X'], x_coords['Y']).values.reshape(self.x_size*self.y_size))
				values.append(vals)

			ret = self.pcas[0].transform(values)
			reshaped = []
			for i in range(self.y_size):
				reshaped.append([])
				for j in range(self.x_size):
					reshaped[i].append(ret)
			reshaped = np.asarray(reshaped)
			coords = {
				'MODE': [k for k in range(self.pcas[0].n_components_)],
				x_coords['X']: X.coords[x_coords['X']].values,
				x_coords['Y']: X.coords[x_coords['Y']].values,
				x_coords['T']: X.coords[x_coords['T']].values,
			}
			data_vars = {'PCS': ([x_coords['Y'], x_coords['X'], x_coords['T'], 'MODE'], reshaped )}
			return xr.Dataset(data_vars, coords=coords)


class SpatioModelPrincipalComponents:
	def __init__(self, n_components=5):
		assert n_components is None or type(n_components) is int, '{} invalid n_components arg, must be None or integer number of components'.format(dt.datetime.now())
		self.n_components = n_components
		self.pca = []

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		X = standardize_dims(X, x_coords)
		varname = [i for i in X.data_vars][0]
		self.x_size, self.y_size = len(X.coords[x_coords['X']].values), len(X.coords[x_coords['Y']].values)
		assert 'M' in x_coords.keys(), '{} couldnt find model dimension on X'.format(dt.datetime.now())
		self.m_size = len(X.coords[x_coords['M']].values)
		values = []
		for j in range(len(X.coords[x_coords['T']].values)):
			iseldict = {x_coords['T']: j}
			vals = np.squeeze(getattr(X, varname).isel(**iseldict).transpose(x_coords['M'], x_coords['Y'], x_coords['X']).values.reshape(self.m_size*self.y_size*self.x_size))
			values.append(vals)
		values = np.asarray(values)
		self.pca = PCA(n_components=self.n_components)
		self.pca.fit(values)

		eofs = []
		eofs_j = self.pca.components_.reshape(self.pca.n_components_, self.y_size*self.x_size*self.m_size )
		for j in range(self.pca.n_components_):
			eofs_i = eofs_j[j].reshape(self.m_size, self.y_size*self.x_size)
			eofs_i = np.asarray([eofs_i[i].reshape(self.x_size, self.y_size).T for i in range(eofs_i.shape[0])])
			eofs.append(eofs_i)
		eofs = np.asarray(eofs)
		coords = {
			'MODE': [k for k in range(self.pca.n_components_)],
			x_coords['X']: X.coords[x_coords['X']].values,
			x_coords['Y']: X.coords[x_coords['Y']].values,
			x_coords['M']: X.coords[x_coords['M']].values,
		}
		data_vars = {'EOFS': ([ 'MODE', x_coords['M'], x_coords['Y'], x_coords['X']], eofs)}
		self.eofs = xr.Dataset(data_vars, coords=coords).EOFS

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		X = standardize_dims(X, x_coords)
		varname = [i for i in X.data_vars][0]
		assert 'M' in x_coords.keys(), '{} couldnt find model dimension on X'.format(dt.datetime.now())
		values = []
		for j in range(len(X.coords[x_coords['T']].values)):
			iseldict = {x_coords['T']: j}
			vals = np.squeeze(getattr(X, varname).isel(**iseldict).transpose(x_coords['M'], x_coords['Y'], x_coords['X']).values.reshape(self.x_size*self.y_size*self.m_size))
			values.append(vals)
		values = np.asarray(values)
		ret = self.pca.transform(values)
		reshaped = []
		for i in range(self.y_size):
			reshaped.append([])
			for j in range(self.x_size):
				reshaped[i].append(ret)
		reshaped = np.asarray(reshaped)
		coords = {
			'MODE': [k for k in range(self.pca.n_components_)],
			x_coords['X']: X.coords[x_coords['X']].values,
			x_coords['Y']: X.coords[x_coords['Y']].values,
			x_coords['T']: X.coords[x_coords['T']].values,
		}
		data_vars = {'PCS': ([x_coords['Y'], x_coords['X'], x_coords['T'], 'MODE'], reshaped )}
		return xr.Dataset(data_vars, coords=coords).transpose()
