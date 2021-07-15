from .pointwise_preprocess import PointWisePreprocess
import numpy as np
import datetime as dt
from ..core import *

class NormalScaler_:
	def __init__(self):
		self.mu, self.std = None, None

	def fit(self, x):
		self.mu = np.nanmean(x)
		self.std = np.nanstd(x)
		self.std = 1 if self.std == 0 else self.std
		self.mu = 0 if self.std == 0 else self.mu

	def transform(self, x):
		assert self.mu is not None and self.std is not None, '{} Must Fit Scaler_ before transform'.format(dt.datetime.now())
		return ((x - self.mu) / self.std)

	def inverse_transform(self, x):
		assert self.mu is not None and self.std is not None, '{} Must Fit Scaler_ before transform'.format(dt.datetime.now())
		return ((x * self.std) + self.mu)


class NormalScaler:
	def __init__(self, **kwargs):
		self.pointwise = []
		self.kwargs = kwargs

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		self.mu = X.mean(x_coords['T'])
		self.std = X.std(x_coords['T'])

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		xvarname = [ i for i in X.data_vars][0]
		return (getattr(X, xvarname) - self.mu ) / self.std

	def inverse_transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		X = standardize_dims(X, x_coords, verbose=verbose)
		xvarname = [ i for i in X.data_vars][0]
		return (getattr(X, xvarname) * self.std) + self.mu

class NormalScalerPointwise:
	def __init__(self, **kwargs):
		self.pointwise = PointWisePreprocess()

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		self.pointwise.fit(X, NormalScaler_, x_coords=x_coords,  verbose=verbose)

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		return self.pointwise.transform(X, x_coords=x_coords,  verbose=verbose)

	def inverse_transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},  verbose=False):
		return self.pointwise.inverse_transform(X, x_coords=x_coords, verbose=verbose)
