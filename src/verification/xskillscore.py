import xskillscore as xs
import xarray as xr
from ..core.utilities import *
import numpy as np
from ..preprocessing.onehot import *

def probabilistic_skill(X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', bn_thresh=0.33, an_thresh=0.67):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim) # X is predictions
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim) # Y is observatiosn

	ohe = RankedTerciles()
	ohe.fit(Y, x_lat_dim=y_lat_dim, x_lon_dim=y_lon_dim, x_sample_dim=y_sample_dim, x_feature_dim=y_feature_dim)
	y_prob = ohe.transform(Y, x_lat_dim=y_lat_dim, x_lon_dim=y_lon_dim, x_sample_dim=y_sample_dim, x_feature_dim=y_feature_dim)
	y_prob= to_xss(y_prob, y_lat_dim, y_lon_dim, y_sample_dim, 'C' )
	preds  = to_xss(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

	y_prob.coords['time'] = preds.coords['time'].values
	y_prob.coords['member'] = preds.coords['member'].values
	y_prob.coords['lat'] = preds.coords['lat'].values
	y_prob.coords['lon'] = preds.coords['lon'].values

	thresholds = np.array([0.0, bn_thresh, an_thresh, 1.0])
	rpss = []
	for i in range(3):
		rpss.append(xs.rps(y_prob.isel(member=i), preds, thresholds, dim=['time']))
	rps = xr.concat(rpss, 'C').assign_coords({'C': [0,1,2]})
	roc = xs.roc(y_prob, preds, bin_edges=np.linspace(0, 1, 11), dim=['time'])
	tercile_contingency = xs.Contingency(y_prob, preds, thresholds, thresholds, dim=['time'])
	rps.name = 'ranked_probability_score'
	roc.name = 'reciever_operating_curve'
	ger = tercile_contingency.gerrity_score()
	ger.name = 'gerrity_score'
	heid = tercile_contingency.heidke_score()
	heid.name = 'heidke_score'
	peri = tercile_contingency.peirce_score()
	peri.name = 'peirce_score'
	acc = tercile_contingency.accuracy()
	acc.name = 'accuracy'
	return xr.merge([rps, roc, ger, heid, peri, acc], compat='override')

def deterministic_skill(X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M'):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim) # X is predictions
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)# Y is observatiosn

	Y1 = to_xss(Y, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim)
	X1 = to_xss(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

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
