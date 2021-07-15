from ..core.data_validation import *
from ..core.utilities import *
import datetime as dt


def pointwise_skill(X, Y, skill_func, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False):
	if 'M' not in x_coords.keys():
		x_coords['M'] = 'M'
	if 'M' not in y_coords.keys():
		y_coords['M'] = 'M'
	X = standardize_dims(X, x_coords, verbose=verbose)
	Y = standardize_dims(Y, y_coords, verbose=verbose)
	x_coords_slice = {'X':x_coords['X'], 'Y':x_coords['Y'], 'T':x_coords['T']}
	y_coords_slice = {'X':y_coords['X'], 'Y':y_coords['Y'], 'T':y_coords['T']}

	check_same_shape(X, Y, x_coords=x_coords_slice, y_coords=y_coords_slice)

	X = X.transpose(x_coords['M'], x_coords['Y'], x_coords['X'], x_coords['T'])
	Y = Y.transpose(y_coords['M'], y_coords['Y'], y_coords['X'], y_coords['T'])
	xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]

	count=0
	if verbose > 1:
		total = len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
		print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

	x_data = getattr(X, xvarname).values
	y_data = getattr(Y, yvarname).values

	results = []
	for ii in range(x_data.shape[0]):
		values = []
		for i in range(x_data.shape[1]):
			values.append([])
			for j in range(x_data.shape[2]):
				#isel_x_dict = {x_coords['M']: ii, x_coords['X']: i, x_coords['Y']: j}
				#isel_y_dict = { y_coords['X']: i, y_coords['Y']: j}
				x_train = x_data[ii, i, j, :]#getattr(X, xvarname).isel(**isel_x_dict).values.reshape((len(X.coords[x_coords['T']].values), 1))
				y_train = y_data[ii, i, j, :]# getattr(Y, yvarname).isel(**isel_y_dict).values.reshape((len(Y.coords[y_coords['T']].values), 1))

				if np.sum(np.isnan(y_train)) > 0 or np.sum(np.isnan(x_train)) > 0:
					values[i].append(np.nan)
				else:
					val = skill_func(np.squeeze(y_train), np.squeeze(x_train))
					values[i].append(val)
				count += 1
				if verbose > 1:
					print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
		res = np.asarray(values)
		results.append(res)

	if verbose > 1:
		print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
	data_vars = {'skill_measure': ([x_coords['M'], x_coords['Y'], x_coords['X']], results)}

	coords = {
		x_coords['M']: X.coords[x_coords['M']],
		x_coords['X']: X.coords[x_coords['X']],
		x_coords['Y']: X.coords[x_coords['Y']]
	}
	return xr.Dataset(data_vars, coords=coords)
