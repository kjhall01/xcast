from ..core.utilities import *

class Normal:
	def __init__(self, **kwargs):
		self.pointwise = []
		self.kwargs = kwargs

	def fit(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim
		self.mu = X.mean(x_sample_dim)
		self.std = X.std(x_sample_dim)
		self.std = self.std.where(self.std != 0, other = 1)


	def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.mu = self.mu.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		self.std = self.std.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })	
		self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim
		r =  (X - self.mu ) / self.std
		r.attrs['generated_by'] =  r.attrs['generated_by'] + '\n  XCAST Normal Transform' if 'generated_by' in r.attrs.keys() else '\n  XCAST Normal Transform'
		return r 

	def inverse_transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
		x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.mu = self.mu.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		self.std = self.std.swap_dims({ self.lat_dim:x_lat_dim, self.lon_dim:x_lon_dim,   self.feature_dim:x_feature_dim}).drop(self.lat_dim).drop(self.lon_dim).drop(self.feature_dim).assign_coords({x_lat_dim: X.coords[x_lat_dim], x_lon_dim:X.coords[x_lon_dim], x_feature_dim: X.coords[x_feature_dim] })
		self.sample_dim, self.lat_dim, self.lon_dim, self.feature_dim = x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim

		ret = []
		for i in range(X.shape[list(X.dims).index(self.feature_dim)]):
			sd = {self.feature_dim: i}
			self.std.coords[self.feature_dim] = [X.coords[self.feature_dim].values[i]]
			self.mu.coords[self.feature_dim] = [X.coords[self.feature_dim].values[i]]
			ret.append( X.isel(**sd) *self.std + self.mu)
		r =  xr.concat(ret, self.feature_dim)
		r.attrs['generated_by'] =  r.attrs['generated_by'] + '\n  XCAST Normal Inverse Transform' if 'generated_by' in r.attrs.keys() else '\n  XCAST Normal Inverse Transform'
		return r 
