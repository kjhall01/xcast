from .individual_wrappers import index_of_agreement_, pearson_p_, pearson_coef_, spearman_p_, spearman_coef_, root_mean_squared_error_, mean_squared_error_, mean_absolute_error_, index_of_agreement_
from .pointwise_skill import *
import xarray as xr

def index_of_agreement(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, index_of_agreement_, x_coords=x_coords, y_coords=y_coords, verbose=verbose,  x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)
	#xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]
	#return xr.apply_ufunc(index_of_agreement_, getattr(X, xvarname), getattr(Y, yvarname), input_core_dims=[[x_coords['T']], [y_coords['T']]])

def pearson_coefficient(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, pearson_coef_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)

def spearman_coefficient(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, spearman_coef_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)

def pearson_significance(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, pearson_p_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)

def spearman_significance(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, spearman_p_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)

def root_mean_squared_error(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, root_mean_squared_error_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)

def mean_squared_error(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, mean_squared_error_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)

def mean_absolute_error(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	return pointwise_skill(X, Y, mean_absolute_error_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks)

def skill(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'},  verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	ioa = index_of_agreement(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure
	pearson_coef = pearson_coefficient(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure
	pearson_p = pearson_significance(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure
	spearman_coef = spearman_coefficient(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure
	spearman_p = spearman_significance(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure
	rmse = root_mean_squared_error(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure
	mse = mean_squared_error(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure
	mae = mean_absolute_error(X, Y,  x_coords=x_coords, y_coords=y_coords, verbose=verbose, x_chunks=x_chunks, y_chunks=y_chunks, t_chunks=t_chunks).skill_measure

	data_vars = {
		'index_of_agreement': ioa,
		'pearson_coefficient': pearson_coef,
		'pearson_p': pearson_p,
		'spearman_coefficient': spearman_coef,
		'root_mean_squared_error': rmse,
		'mean_squared_error': mse,
		'mean_absolute_error': mae
	}
	return xr.Dataset(data_vars, coords=ioa.coords)
