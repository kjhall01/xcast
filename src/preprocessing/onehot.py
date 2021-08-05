import scipy.stats as ss
from ..core.utilities import *
import numpy as np

class NormalTerciles:
	def __init__(self, normal_width=0.34):
		self.low_thresh, self.high_thresh = ss.norm(0, 1).interval(normal_width)

	def fit(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = X.isel()
		self.mu = X1.mean(x_sample_dim)
		self.std = X1.std(x_sample_dim)

	def transform(self,X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = (X - self.mu ) / self.std

		X_BN = X1.where(X1 < self.low_thresh, other=-999)
		X_BN = X_BN.where(X_BN == -999, other=1.0)
		X_BN = X_BN.where(X_BN == 1.0, other= 0)


		X_AN = X1.where(X1 > self.high_thresh, other=-998)
		X_AN = X_AN.where(X_AN == -998, other=1.0)
		X_AN = X_AN.where(X_AN == 1.0, other= 0)

		X_N = X1.where(self.low_thresh <= X1, other = 0.0)
		X_N = X_N.where(X_N <= self.high_thresh, other=0.0)
		X_N = X_N.where(X_N == 0.0, other=1.0)
		X1 = xr.concat([X_BN, X_N, X_AN], 'C')
		return X1.assign_coords({'C': [0,1,2]})

class RankedTerciles:
	def __init__(self, low_thresh=0.33, high_thresh=0.67):
		self.low_thresh, self.high_thresh = low_thresh, high_thresh

	def fit(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		iseldict = {x_feature_dim: 0}
		X1 = X.isel(**iseldict)
		self.high_threshold = X1.reduce(np.nanpercentile, dim=x_sample_dim, q=self.high_thresh)
		self.low_threshold = X1.reduce(np.nanpercentile, dim=x_sample_dim, q=self.low_thresh)

	def transform(self,X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M'):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		iseldict = {x_feature_dim: 0}
		X1 = X.isel(**iseldict)

		X_BN = X1.where(X1 < self.low_threshold, other=-999)
		X_BN = X_BN.where(X_BN == -999, other=1.0)
		X_BN = X_BN.where(X_BN == 1.0, other= 0)


		X_AN = X1.where(X1 > self.high_threshold, other=-998)
		X_AN = X_AN.where(X_AN == -998, other=1.0)
		X_AN = X_AN.where(X_AN == 1.0, other= 0)

		X_N = X1.where(self.low_threshold <= X1, other = 0.0)
		X_N = X_N.where(X_N <= self.high_threshold, other=0.0)
		X_N = X_N.where(X_N == 0.0, other=1.0)
		X1 = xr.concat([X_BN, X_N, X_AN], 'C')
		return X1.assign_coords({'C': [0,1,2]})
