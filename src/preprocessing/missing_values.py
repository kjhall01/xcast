import numpy as np
from ..core.utilities import *
import xarray as xr

def fill_constant(X, val, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	X1 = X.where(X != missing_value, other=np.nan)
	return X1.fillna(val)

def fill_time_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	X1 = X.where(X != missing_value, other=np.nan)
	filled = X1.fillna(X1.mean(x_sample_dim))
	assert np.isnan(filled).sum().values == 0, 'Found empty space/feature slice- at least one point in at least one feature is missing data in all samples '
	return filled

def fill_space_mean(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', missing_value=np.nan):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	X1 = X.where(X != missing_value, other=np.nan)
	filled = X1.fillna(X1.mean(dim=[x_lat_dim, x_lon_dim]))
	assert np.isnan(filled).sum().values == 0, 'Found empty sample/feature slice - at least one feature for at least one sample is missing all data across lat/lon '
	return filled
