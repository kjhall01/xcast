from ..core import *
import scipy.stats as ss
import xarray as xr

class ProbabilisticScaler:
	def __init__(self, normal_width=0.34, **kwargs):
		self.low_thresh, self.high_thresh = ss.norm(0, 1).interval(normal_width)

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
		X = (getattr(X, xvarname) - self.mu ) / self.std

		X_BN = X.where(getattr(X, xvarname) < self.low_thresh, other=-999)
		X_BN = X_BN.where(getattr(X_BN, xvarname) == -999, other=1.0)
		X_BN = X_BN.where(getattr(X_BN, xvarname) == 1.0, other= 0)


		X_AN = X.where(getattr(X, xvarname) > self.high_thresh, other=-998)
		X_AN = X_AN.where(getattr(X_AN, xvarname) == -998, other=1.0)
		X_AN = X_AN.where(getattr(X_AN, xvarname) == 1.0, other= 0)

		X_N = X.where(self.low_thresh < getattr(X, xvarname), other = 0.0)
		X_N = X_N.where(getattr(X_N, xvarname) < self.high_thresh, other=0.0)
		X_N = X_N.where(getattr(X_N, xvarname) == 0.0, other=1.0)
		X = xr.concat([X_BN, X_N, X_AN], 'C')
		return X.assign_coords({'C': [0,1,2]})
