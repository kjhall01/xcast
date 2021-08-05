import numpy as np
from ..core.utilities import *

def fill_constant(X, val, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	X1 = X.where(X != missing_value, other=np.nan)
	return X1.fillna(val)

def fill_time_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	X1 = X.where(X != missing_value, other=np.nan)
	return X1.fillna(X1.mean(x_sample_dim))

def fill_space_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	X1 = X.where(X != missing_value, other=np.nan)
	return X1.fillna(X1.mean(x_lat_dim).mean(x_lon_dim))
