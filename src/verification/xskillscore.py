#import xskillscore as xs
import xarray as xr
from ..core.utilities import *
import numpy as np

def _chunk(X, Y, x_lat_dim, x_lon_dim, y_lat_dim, y_lon_dim, x_sample_dim, x_feature_dim, y_sample_dim, y_feature_dim, lat_chunks, lon_chunks):
	X1 = X.chunk({x_lat_dim: max( len(X.coords[x_lat_dim].values)// lat_chunks,1), x_lon_dim: max(len(X.coords[x_lon_dim].values)// lon_chunks,1), x_sample_dim:len(X.coords[x_sample_dim].values) , x_feature_dim:1})
	Y1 = Y.chunk({y_lat_dim: max(len(Y.coords[y_lat_dim].values) // lat_chunks, 1), y_lon_dim: max(len(Y.coords[y_lon_dim].values) // lon_chunks,1), y_sample_dim:len(Y.coords[y_sample_dim].values), y_feature_dim:1})
	return X1, Y1

def xskillscore_deterministic(X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', lat_chunks=1, lon_chunks=1,  y_sample_dim='T', y_feature_dim='M'):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim) # X is predictions
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)# Y is observatiosn

	X1, Y1 = _chunk(X, Y, x_lat_dim, x_lon_dim, y_lat_dim, y_lon_dim, x_sample_dim, x_feature_dim, y_sample_dim, y_feature_dim, lat_chunks, lon_chunks)

	Y1 = to_xss(Y1, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim)
	X1 = to_xss(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

	Y1.coords['time'] = X1.coords['time'].values
	Y1.coords['lat'] = X1.coords['lat'].values
	Y1.coords['lon'] = X1.coords['lon'].values


	pco = xs.pearson_r(Y1, X1, dim='time')
	pco.name = 'pearson_coefficient'

	sco = xs.spearman_r(Y1, X1, dim='time')
	sco.name = 'spearman_coefficient'

	ess = xs.effective_sample_size(Y1, X1, dim='time')
	ess.name = 'effective_sample_size'

	pre = xs.pearson_r_eff_p_value(Y1, X1, dim='time')
	pre.name = 'pearson_effective_p_value'

	prp = xs.pearson_r_p_value(Y1, X1, dim='time')
	prp.name = 'pearson_p_value'

	lsl = xs.linslope(Y1, X1, dim='time')
	lsl.name = 'slope_linear_fit'

	sre = xs.spearman_r_eff_p_value(Y1, X1, dim='time')
	sre.name = 'spearman_effective_p_value'

	srp = xs.spearman_r_p_value(Y1, X1, dim='time')
	srp.name = 'spearman_p_value'

	r2 = xs.r2(Y1, X1, dim='time')
	r2.name = 'determination_coefficient'

	mae = xs.mae(Y1, X1, dim='time')
	mae.name = 'mean_absolute_error'

	mape = xs.mape(Y1, X1, dim='time')
	mape.name = 'mean_absolute_percentage_error'

	me = xs.me(Y1, X1, dim='time')
	me.name= 'mean_error'

	mse = xs.mse(Y1, X1, dim='time')
	mse.name = 'mean_squared_error'

	med = xs.median_absolute_error(Y1, X1, dim='time')
	med.name = 'median_absolute_error'

	rmse = xs.rmse(Y1, X1, dim='time')
	rmse.name = 'root_mean_squared_error'

	smape = xs.smape(Y1, X1, dim='time')
	smape.name = 'symmetric_mean_absolute_percentage_error'

	return xr.merge([smape, rmse, med, mse, me ,mape, mae, r2, srp , sre, lsl, prp, pre, ess, sco, pco ])
