from ..core.utilities import *

class Normal:
	def __init__(self, **kwargs):
		self.pointwise = []
		self.kwargs = kwargs

	def fit(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.mu = X.mean(x_sample_dim)
		self.std = X.std(x_sample_dim)
		self.std = self.std.where(self.std != 0, other = 1)
		self.feature_dim = x_feature_dim


	def transform(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return (X - self.mu ) / self.std

	def inverse_transform(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		ret = []
		for i in range(X.shape[list(X.dims).index(x_feature_dim)]):
			sd = {x_feature_dim: i}
			self.std.coords[self.feature_dim] = [X.coords[x_feature_dim].values[i]]
			self.mu.coords[self.feature_dim] = [X.coords[x_feature_dim].values[i]]
			ret.append( X.isel(**sd) *self.std + self.mu)
		return  xr.concat(ret, x_feature_dim)
