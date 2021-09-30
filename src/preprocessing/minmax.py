from ..core.utilities import *

class MinMax:
	def __init__(self, min=-1, max=1):
		self.range_min, self.range_max = min, max
		self.range = max - min
		self.min, self.max, self.x_range = None, None, None

	def fit(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim
		X1 = X.isel()
		self.min = X1.min(x_sample_dim)
		self.max = X1.max(x_sample_dim)
		self.x_range = self.max - self.min
		self.x_range = self.x_range.where(self.x_range != 0, other=1)

	def transform(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#X1 = X.rename({x_lat_dim: self.lat_dim, x_lon_dim: self.lon_dim, x_sample_dim: self.sample_dim})
		self.min = self.min.rename({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim})
		self.max = self.max.rename({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim})
		self.x_range = self.max.rename({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim})
		self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim

		assert self.min is not None and self.max is not None, '{} Must Fit MinMaxScaler before transform'.format(dt.datetime.now())
		return ((X - self.min) / self.x_range) * self.range + self.range_min

	def inverse_transform(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		assert self.min is not None and self.max is not None, '{} Must Fit MinMaxScaler before inverse transform'.format(dt.datetime.now())
		self.min = self.min.rename({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim})
		self.max = self.max.rename({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim})
		self.x_range = self.max.rename({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim})
		self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim

		ret = []
		for i in range(X.shape[list(X.dims).index(self.feature_dim)]):
			sd = {x_feature_dim: i}
			self.max.coords[self.feature_dim] = [X.coords[self.feature_dim].values[i]]
			self.min.coords[self.feature_dim] = [X.coords[self.feature_dim].values[i]]
			self.x_range.coords[self.feature_dim] = [X.coords[self.feature_dim].values[i]]
			ret.append(((X.isel(**sd) - self.range_min) / self.range) * self.x_range + self.min)
		return  xr.concat(ret, self.feature_dim)
