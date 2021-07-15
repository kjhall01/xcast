from .pointwise_preprocess import PointWisePreprocess
import numpy as np
import datetime as dt
from ..core import *

class MinMaxScaler_:
	def __init__(self, min=-1, max=1, **kwargs):
		self.range_min, self.range_max = min, max
		self.range = max - min
		self.min, self.max, self.x_range = None, None, None

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False):
		self.min = X.min(x_coords['T'])
		self.max = X.max(x_coords['T'])
		self.x_range = self.max - self.min

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False):
		return ((X - self.min) / self.x_range) * self.range + self.range_min

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False):
		return ((X - self.range_min) / self.range) * self.x_range + self.min

class MinMaxScaler:
	def __init__(self, min=-1, max=1 , **kwargs):
		self.range_min, self.range_max = min, max
		self.range = max - min
		self.min, self.max, self.x_range = None, None, None

	def fit(self, x, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		x = standardize_dims(x, x_coords, verbose=verbose)

		self.min = np.nanmin(x)
		self.max = np.nanmax(x)
		self.x_range = self.max - self.min

	def transform(self, x, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		x = standardize_dims(x, x_coords, verbose=verbose)
		xvarname = [i for i in x.data_vars][0]
		assert self.min is not None and self.max is not None, '{} Must Fit MinMaxScaler_ before transform'.format(dt.datetime.now())
		return ((getattr(x, xvarname) - self.min) / self.x_range) * self.range + self.range_min

	def inverse_transform(self, x, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		if 'M' not in x_coords.keys():
			x_coords['M'] = 'M'
		x = standardize_dims(x, x_coords, verbose=verbose)
		xvarname = [i for i in x.data_vars][0]
		assert self.mu is not None and self.std is not None, '{} Must Fit Scaler_ before transform'.format(dt.datetime.now())
		return ((getattr(x, xvarname) - self.range_min) / self.range) * self.x_range + self.min

class MinMaxScalerPointwise:
	def __init__(self, min=-1, max=1, **kwargs):
		self.min, self.max = min, max
		self.pointwise = PointWisePreprocess(min=min, max=max)

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False):
		self.pointwise.fit(X, MinMaxScaler_, x_coords=x_coords,  verbose=verbose)

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		return self.pointwise.transform(X, x_coords=x_coords,  verbose=verbose)

	def inverse_transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		return self.pointwise.inverse_transform(X, x_coords=x_coords, verbose=verbose)
