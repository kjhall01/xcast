from ..core.data_validation import *
from ..core.utilities import *
import uuid
from pathlib import Path
import xarray as xr
import datetime as dt

def pointwise_skill(X, Y, skill_func, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
	tf_uuid = str(uuid.uuid4())
	ids = []
	if 'M' not in x_coords.keys():
		x_coords['M'] = 'M'
	X = standardize_dims(X, x_coords, verbose=verbose)
	X = X.chunk({x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values)/t_chunks)), x_coords['M']: len(X.coords[x_coords['M']].values), x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks)), x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks))  })

	if 'M' not in y_coords.keys():
		y_coords['M'] = 'M'
	Y = standardize_dims(Y, y_coords, verbose=verbose)
	Y = Y.chunk({y_coords['T']: max(1, int(len(Y.coords[y_coords['T']].values)/t_chunks)), y_coords['M']: len(Y.coords[y_coords['M']].values), y_coords['Y']: max(1, int(len(Y.coords[y_coords['Y']].values) / y_chunks)), y_coords['X']: max(1, int(len(Y.coords[y_coords['X']].values) / x_chunks))  })

	Y = Y.transpose(y_coords['M'], y_coords['Y'], y_coords['X'], y_coords['T'])
	X = X.transpose(x_coords['M'], x_coords['Y'], x_coords['X'], x_coords['T'])
	count=0
	if verbose > 1:
		total = len(X.chunks[x_coords['M']]) * len(X.chunks[x_coords['Y']]) * len(X.chunks[x_coords['X']])
		print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

	for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
		indx = 0
		for ii in range(i):
			indx += X.chunks[x_coords['Y']][ii]
		for j in range(len(X.chunks[x_coords['X']])): # chunks in X
			jndx = 0
			for jj in range(j):
				jndx += X.chunks[x_coords['X']][jj]
			iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
			pointwise_skill_one(X.isel(**iseldict), Y.isel(**iseldict), skill_func, x_coords=x_coords, y_coords=y_coords,  verbose=verbose-1).to_netcdf(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
			ids.append(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
			count += 1
			if verbose > 1:
				print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
	if verbose > 1:
		print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
	return xr.open_mfdataset(ids, chunks='auto', parallel=True)


def pointwise_skill_one(X, Y, skill_func, x_coords={'X':'X', 'Y':'Y', 'T':'T'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False):

	X = standardize_dims(X, x_coords, verbose=verbose)
	Y = standardize_dims(Y, y_coords, verbose=verbose)
	check_same_shape(X, Y, x_coords=x_coords, y_coords=y_coords)
	xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]
	#X = X.fillna(X.mean())
	#Y = Y.fillna(Y.mean())
	x_data = getattr(X, xvarname).values
	y_data = getattr(Y, yvarname).values
	if 'M' in x_coords.keys():
		count=0
		if verbose > 2:
			total = len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
			print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		results = []
		for ii in range(len(X.coords[x_coords['M']].values)):
			values = []
			for i in range(len(X.coords[x_coords['Y']].values)):
				values.append([])
				for j in range(len(X.coords[x_coords['X']].values)):
					#isel_x_dict = {x_coords['M']: ii, x_coords['X']: i, x_coords['Y']: j}
					#isel_y_dict = { y_coords['X']: i, y_coords['Y']: j}
					x_train = x_data[ii, i, j, :] #getattr(X, xvarname).isel(**isel_x_dict).values.reshape((len(X.coords[x_coords['T']].values), 1))
					if len(x_train.shape) < 2:
						x_train = x_train.reshape(-1,1)
					if x_train.shape[1] > x_train.shape[0]:
						x_train = x_train.T
					y_train = y_data[ii, i, j , :] #getattr(Y, yvarname).isel(**isel_y_dict).values.reshape((len(Y.coords[y_coords['T']].values), 1))
					if len(y_train.shape) < 2:
						y_train = y_train.reshape(-1,1)
					if y_train.shape[1] > y_train.shape[0]:
						y_train = y_train.T
					if np.sum(np.isnan(y_train)) > 0 or np.sum(np.isnan(x_train)) > 0:
						values[i].append(np.nan)
					else:
						val = skill_func(np.squeeze(y_train), np.squeeze(x_train))
						values[i].append(val)
					count += 1
					if verbose > 2:
						print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
			res = np.asarray(values)
			results.append(res)

		if verbose > 2:
			print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))
		data_vars = {'skill_measure': ([x_coords['M'], x_coords['Y'], x_coords['X']], results)}

		coords = {
			x_coords['M']: X.coords[x_coords['M']],
			x_coords['X']: X.coords[x_coords['X']],
			x_coords['Y']: X.coords[x_coords['Y']]
		}
		return xr.Dataset(data_vars, coords=coords)
	else:
		count=0
		if verbose > 2:
			total = len(X.coords[x_coords['Y']].values) * len(X.coords[x_coords['X']].values)
			print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		values = []
		for i in range(len(X.coords[x_coords['X']].values)):
			values.append([])
			for j in range(len(X.coords[x_coords['Y']].values)):
				#isel_x_dict = {x_coords['X']: i, x_coords['Y']: j}
				#isel_y_dict = { y_coords['X']: i, y_coords['Y']: j}
				x_train = x_data[i, j, :] #getattr(X, xvarname).isel(**isel_x_dict).values.reshape((len(X.coords[x_coords['T']].values), 1))
				if len(x_train.shape) < 2:
					x_train = x_train.reshape(-1,1)
				if x_train.shape[1] > x_train.shape[0]:
					x_train = x_train.T
				y_train = y_data[ i, j , :] #getattr(Y, yvarname).isel(**isel_y_dict).values.reshape((len(Y.coords[y_coords['T']].values), 1))
				if len(y_train.shape) < 2:
					y_train = y_train.reshape(-1,1)
				if y_train.shape[1] > y_train.shape[0]:
					y_train = y_train.T
				if np.sum(np.isnan(y_train)) > 0 or np.sum(np.isnan(x_train)) > 0:
					values[i].append(np.nan)
				else:
					val = skill_func(np.squeeze(y_train), np.squeeze(x_train))
					values[i].append(val)
				count += 1
				if verbose > 2:
					print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
		results = np.asarray(values)
		if verbose > 2:
			print('{} PointWiseSkill Eval for {}: ['.format(dt.datetime.now(), skill_func.__name__) + '*'*25 +'] 100% ({}/{})'.format( total, total))

		data_vars = {'skill_measure': ([ x_coords['Y'], x_coords['X']], results.T)}
		coords = {
			x_coords['X']: X.coords[x_coords['X']],
			x_coords['Y']: X.coords[x_coords['Y']]
		}
		return xr.Dataset(data_vars, coords=coords)
