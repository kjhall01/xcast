from ..core.utilities import *

class MinMax:
	def __init__(self, min=-1, max=1):
		self.range_min, self.range_max = min, max
		self.range = max - min
		self.min, self.max, self.x_range = None, None, None

	def fit(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim
		X1 = X.isel()
		self.min = X1.min(x_sample_dim)
		self.max = X1.max(x_sample_dim)
		self.x_range = self.max - self.min
		self.x_range = self.x_range.where(self.x_range != 0, other=1)

	def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#X1 = X.swap_dims({x_lat_dim: self.lat_dim, x_lon_dim: self.lon_dim, x_sample_dim: self.sample_dim})
		mn = self.min.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		mx = self.max.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		x_range = self.x_range.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		#self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim

		assert self.min is not None and self.max is not None, '{} Must Fit MinMaxScaler before transform'.format(dt.datetime.now())
		r =  ((X - mn) / x_range) * self.range + self.range_min
		r.attrs['generated_by'] =  r.attrs['generated_by'] + '\n  XCAST MinMax  Transform' if 'generated_by' in r.attrs.keys() else '\n  XCAST MinMax  Transform'
		return r

	def inverse_transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		assert self.min is not None and self.max is not None, '{} Must Fit MinMaxScaler before inverse transform'.format(dt.datetime.now())
		mn = self.min.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		mx = self.max.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		x_range = self.x_range.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })

		r = ((X - self.range_min) / self.range) * x_range + mn
		r.attrs['generated_by'] =  r.attrs['generated_by'] + '\n  XCAST MinMax Inverse Transform' if 'generated_by' in r.attrs.keys() else '\n  XCAST MinMax Inverse Transform'
		return r

