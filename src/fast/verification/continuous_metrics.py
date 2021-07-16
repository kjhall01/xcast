from .individual_wrappers import index_of_agreement_, pearson_p_, pearson_coef_, spearman_p_, spearman_coef_, root_mean_squared_error_, mean_squared_error_, mean_absolute_error_, index_of_agreement_, roc_auc_score_, average_precision_score_
from .pointwise_skill import *

def index_of_agreement(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, index_of_agreement_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def pearson_coefficient(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, pearson_coef_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def spearman_coefficient(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, spearman_coef_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def pearson_significance(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, pearson_p_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def spearman_significance(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, spearman_p_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def root_mean_squared_error(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, root_mean_squared_error_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def mean_squared_error(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, mean_squared_error_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def mean_absolute_error(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	return pointwise_skill(X, Y, mean_absolute_error_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def area_under_roc(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'C':'C'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'C':'C'},  verbose=False):
	return pointwise_skill_proba(X, Y, roc_auc_score_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def precision(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'C':'C'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'C':'C'},  verbose=False):
	return pointwise_skill_proba(X, Y, average_precision_score_, x_coords=x_coords, y_coords=y_coords, verbose=verbose)

def skill_proba(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'C':'C'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'C':'C'},  verbose=False):
	roc = area_under_roc(X, Y, x_coords=x_coords, y_coords=y_coords,  verbose=False)
	prec = precision(X, Y, x_coords=x_coords, y_coords=y_coords,  verbose=False)

	data_vars = {
		'roc': roc.skill_measure,
		'precision': prec.skill_measure,
	}
	return xr.Dataset(data_vars, coords=roc.coords)

def skill(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False):
	ioa = index_of_agreement(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)
	pearson_coef = pearson_coefficient(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)
	pearson_p = pearson_significance(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)
	spearman_coef = spearman_coefficient(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)
	spearman_p = spearman_significance(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)
	rmse = root_mean_squared_error(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)
	mse = mean_squared_error(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)
	mae = mean_absolute_error(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose)

	data_vars = {
		'index_of_agreement': ioa.skill_measure,
		'pearson_coefficient': pearson_coef.skill_measure,
		'pearson_p': pearson_p.skill_measure,
		'spearman_coefficient': spearman_coef.skill_measure,
		'root_mean_squared_error': rmse.skill_measure,
		'mean_squared_error': mse.skill_measure,
		'mean_absolute_error': mae.skill_measure
	}
	return xr.Dataset(data_vars, coords=ioa.coords)
