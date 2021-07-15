from ..core import *
from scipy.interpolate import interp2d
import numpy as np
from sklearn.impute import KNNImputer
import datetime as dt

def regrid(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_value=-999, interp='linear', fill='mean', verbose=False):
	if 'M' not in x_coords.keys():
		x_coords['M'] = 'M'
	if 'M' not in y_coords.keys():
		y_coords['M'] = x_coords['M'] # coerce Y onto X's M dim

	X = standardize_dims(X, x_coords , verbose=verbose)
	Y = standardize_dims(Y, y_coords, verbose=verbose)

	if are_same_shape(X, Y, {'X':x_coords['X'], 'Y':x_coords['Y']}, {'X':y_coords['X'], 'Y':y_coords['Y']}):
		return X
	X = X.transpose(x_coords['T'], x_coords['M'], x_coords['Y'], x_coords['X'])
	Y = Y.transpose(y_coords['T'], y_coords['M'], y_coords['Y'], y_coords['X'])


	xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]

	if fill == 'kernel':
		X2 = kernel_average(X, x_coords, weighting=2, verbose=verbose , missing_value = missing_value)
		Y = kernel_average(Y, y_coords, weighting=2, verbose=verbose, missing_value = missing_value)
	elif fill == 'mean':
		X2 = X.fillna(X.mean(x_coords['X']).mean(x_coords['Y']))
		Y = Y.fillna(Y.mean(y_coords['X']).mean(y_coords['Y']))
	elif fill == 'knn':
		X2 = knn(X, x_coords, verbose=verbose)
		Y = knn(Y, y_coords, verbose=verbose)
	else:
		assert False, '{} fill must be one of [knn, mean, kernel]'.format(dt.datetime.now())

	X2 = X2.transpose(x_coords['M'], x_coords['T'], x_coords['Y'], x_coords['X'])
	Y = Y.transpose(y_coords['M'], y_coords['T'], y_coords['Y'], y_coords['X'])

	x_vals, y_vals = X2.coords[x_coords['X']].values, X2.coords[x_coords['Y']].values
	x_vals2, y_vals2 = Y.coords[y_coords['X']].values, Y.coords[y_coords['Y']].values

	x_data, y_data = getattr(X2, xvarname).values, getattr(Y, yvarname).values
	regridded = []
	for i in range(x_data.shape[0]):
		regridded.append([])
		for j in range(x_data.shape[1]):
			#iseldict = {x_coords['T']: j, x_coords['M']: i}
			vals = x_data[i, j, :, :]#getattr(X2, xvarname).isel(**iseldict)
			interp_func  = interp2d(x_vals, y_vals, vals, kind=interp)

			#iseldict = {y_coords['T']: j, y_coords['M']: 0}
			#vals2 = getattr(Y, yvarname).isel(**iseldict)
			interped = interp_func(x_vals2, y_vals2)
			regridded[i].append(interped )
	regridded = np.asarray(regridded)
	coords = {
		x_coords['T']: [i for i in X2.coords[x_coords['T']].values],
		x_coords['M']: [i for i in X2.coords[x_coords['M']].values],
		x_coords['X']: [i for i in Y.coords[y_coords['X']].values],
		x_coords['Y']: [i for i in Y.coords[y_coords['Y']].values]
	}
	data_vars = { xvarname: ([x_coords['M'], x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
	return xr.Dataset(data_vars, coords=coords)


def _kernel_average(x, missing_value=-999, weighting=0):
	if len(x.shape) == 1:
		x = x.reshape(-1, 1)
	assert len(x.shape) == 2, '{} wrong shape for _kernel_average- {}'.format(dt.datetime.now(), x.shape)
	x = x.copy()
	x[np.isnan(x)] = missing_value
	blah = (x == missing_value)
	x_new = np.pad(x, 1, constant_values=missing_value)
	x_new[x_new == missing_value] = np.nan
	x_tl = np.asarray([[x_new[i-1,j-1] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)]) / np.sqrt(2)**weighting
	x_tm = np.asarray([[x_new[i-1,j] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)])
	x_tr = np.asarray([[x_new[i-1,j+1] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)]) / np.sqrt(2)**weighting
	x_ml = np.asarray([[x_new[i,j-1] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)])
	x_mr = np.asarray([[x_new[i,j+1] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)])
	x_bl = np.asarray([[x_new[i+1,j-1] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)]) / np.sqrt(2)**weighting
	x_bm = np.asarray([[x_new[i+1,j] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)])
	x_br = np.asarray([[x_new[i+1,j+1] for j in range(1, x_new.shape[1]-1)] for i in range(1, x_new.shape[0]-1)]) / np.sqrt(2)**weighting
	data = np.asarray([x_tl, x_tm, x_tr, x_ml, x_mr, x_bl, x_bm, x_br])
	new = np.nanmean(data, axis=0)
	new[~blah] = x[~blah]
	return new

def kernel_average(X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_value=-999, weighting=0, verbose=False):

	X = standardize_dims(X, x_coords, verbose=verbose)
	xvarname = [i for i in X.data_vars][0]

	if 'M' in x_coords.keys():
		regridded = []
		count = 0
		for i in range(len(X.coords[x_coords['M']].values)):
			regridded.append([])
			for j in range(len(X.coords[x_coords['T']].values)):
				iseldict = {x_coords['T']: j, x_coords['M']: i}
				vals = getattr(X, xvarname).isel(**iseldict).values
				nmissing = np.sum(np.isnan(vals))
				assert nmissing < vals.shape[0] * vals.shape[1], '{} cannot kernel average a fully missing slice'.format(dt.datetime.now())
				while nmissing > 0:
					vals = _kernel_average(vals, weighting=weighting, missing_value=missing_value)
					nmissing = np.sum(np.isnan(vals))
				regridded[i].append( vals )
				count += 1
				if count % 10 == 1 and verbose > 1:
					pct = int( 25* count / (len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['T']].values)))
					print('{} Kernel Average: ['.format(dt.datetime.now()) +  '*'*pct + ' '*(25 - pct) + '] {}% ({}/{})'.format(int(count / (len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['T']].values))*100), count, (len(X.coords[x_coords['M']].values) * len(X.coords[x_coords['T']].values))), end='\r')
		if verbose > 1:
			print()


		regridded = np.asarray(regridded)
		coords = {
			x_coords['T']: [i for i in X.coords[x_coords['T']].values],
			x_coords['M']: [i for i in X.coords[x_coords['M']].values],
			x_coords['X']: [i for i in X.coords[x_coords['X']].values],
			x_coords['Y']: [i for i in X.coords[x_coords['Y']].values]
		}
		data_vars = { xvarname: ([x_coords['M'], x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
		return xr.Dataset(data_vars, coords=coords)
	else:
		count=0
		regridded = []
		for j in range(len(X.coords[x_coords['T']].values)):
			iseldict = {x_coords['T']: j}
			vals = getattr(X, xvarname).isel(**iseldict).values
			nmissing = np.sum(np.isnan(vals))
			while nmissing > 0:
				vals = _kernel_average(vals, weighting=weighting, missing_value=missing_value)
				nmissing = np.sum(np.isnan(vals))

			regridded.append( vals )
			count += 1
			if count % 10 == 1 and verbose > 1:
				pct = int( 25* count / (len(X.coords[x_coords['T']].values)))
				print('{} Kernel Average: ['.format(dt.datetime.now()) +  '*'*pct + ' '*(25 - pct) + '] {}% ({}/{})'.format(int(count / (len(X.coords[x_coords['T']].values))*100), count, ( len(X.coords[x_coords['T']].values))), end='\r')
		if verbose > 1:
			print()
		regridded = np.asarray(regridded)
		coords = {
			x_coords['T']: [i for i in X.coords[x_coords['T']].values],
			x_coords['X']: [i for i in X.coords[x_coords['X']].values],
			x_coords['Y']: [i for i in X.coords[x_coords['Y']].values]
		}
		data_vars = { xvarname: ([ x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
		return xr.Dataset(data_vars, coords=coords)


def knn(X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_values=np.nan, weights='distance', knn=2, by_year=True, verbose=False):
	X = standardize_dims(X, x_coords, verbose=verbose)
	xvarname = [i for i in X.data_vars][0]
	if by_year:
		if 'M' in x_coords.keys():
			regridded = []
			for i in range(len(X.coords[x_coords['M']].values)):
				regridded.append([])
				for j in range(len(X.coords[x_coords['T']].values)):
					iseldict = {x_coords['T']: j, x_coords['M']: i}
					vals = getattr(X, xvarname).isel(**iseldict).stack(z=(x_coords['X'], x_coords['Y']))
					xvals = np.asarray([n[0] for n in vals.coords['z'].values]).reshape(-1, 1)
					yvals = np.asarray([n[1] for n in vals.coords['z'].values]).reshape(-1, 1)
					c = np.hstack((xvals, yvals))
					values = np.hstack((c, vals.values.reshape(-1,1)))
					imputer = KNNImputer(missing_values=missing_values, n_neighbors=knn, weights=weights )
					filled = imputer.fit_transform(values)[:, 2].reshape(len(X.coords[x_coords['Y']].values), len(X.coords[x_coords['X']].values))
					regridded[i].append( filled )
			regridded = np.asarray(regridded)
			coords = {
				x_coords['T']: [i for i in X.coords[x_coords['T']].values],
				x_coords['M']: [i for i in X.coords[x_coords['M']].values],
				x_coords['X']: [i for i in X.coords[x_coords['X']].values],
				x_coords['Y']: [i for i in X.coords[x_coords['Y']].values]
			}
			data_vars = { xvarname: ([x_coords['M'], x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
			return xr.Dataset(data_vars, coords=coords)
		else:
			regridded = []
			for j in range(len(X.coords[x_coords['T']].values)):
				iseldict = {x_coords['T']: j}
				vals = getattr(X, xvarname).isel(**iseldict).stack(z=(x_coords['X'], x_coords['Y']))
				xvals = np.asarray([n[0] for n in vals.coords['z'].values]).reshape(-1, 1)
				yvals = np.asarray([n[1] for n in vals.coords['z'].values]).reshape(-1, 1)
				c = np.hstack((xvals, yvals))
				values = np.hstack((c, vals.values.reshape(-1,1)))
				imputer = KNNImputer(missing_values=missing_values, n_neighbors=knn, weights=weights )
				filled = imputer.fit_transform(values)[:, 2].reshape(len(X.coords[x_coords['Y']].values), len(X.coords[x_coords['X']].values))
				regridded.append( filled )
			regridded = np.asarray(regridded)
			coords = {
				x_coords['T']: [i for i in X.coords[x_coords['T']].values],
				x_coords['X']: [i for i in X.coords[x_coords['X']].values],
				x_coords['Y']: [i for i in X.coords[x_coords['Y']].values]
			}
			data_vars = { xvarname: ([ x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
			return xr.Dataset(data_vars, coords=coords)
	else:
		if 'M' in x_coords.keys():
			regridded = []
			for i in range(len(X.coords[x_coords['M']].values)):
				iseldict = { x_coords['M']: i}
				vals = getattr(X, xvarname).isel(**iseldict).stack(z=(x_coords['X'], x_coords['Y'], x_coords['T']))
				xvals = np.asarray([n[0] for n in vals.coords['z'].values]).reshape(-1, 1)
				yvals = np.asarray([n[1] for n in vals.coords['z'].values]).reshape(-1, 1)
				tvals = np.asarray([ii for ii in range(len(vals.coords['z'].values))]).reshape(-1, 1)
				c = np.hstack((xvals, yvals, tvals))
				values = np.hstack((c, vals.values.reshape(-1,1)))
				imputer = KNNImputer(missing_values=missing_values, n_neighbors=knn, weights=weights )
				filled = imputer.fit_transform(values)[:, 3].reshape(len(X.coords[x_coords['Y']].values), len(X.coords[x_coords['X']].values), len(X.coords[x_coords['T']].values))
				regridded.append( filled )
			regridded = np.asarray(regridded)
			coords = {
				x_coords['T']: [i for i in X.coords[x_coords['T']].values],
				x_coords['M']: [i for i in X.coords[x_coords['M']].values],
				x_coords['X']: [i for i in X.coords[x_coords['X']].values],
				x_coords['Y']: [i for i in X.coords[x_coords['Y']].values]
			}
			data_vars = { xvarname: ([x_coords['M'], x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
			return xr.Dataset(data_vars, coords=coords)
		else:
			vals = getattr(X, xvarname).stack(z=(x_coords['X'], x_coords['Y'], x_coords['T']))
			xvals = np.asarray([n[0] for n in vals.coords['z'].values]).reshape(-1, 1)
			yvals = np.asarray([n[1] for n in vals.coords['z'].values]).reshape(-1, 1)
			tvals = np.asarray([ii for ii in range(len(vals.coords['z'].values))]).reshape(-1, 1)
			c = np.hstack((xvals, yvals, tvals))
			values = np.hstack((c, vals.values.reshape(-1,1)))
			imputer = KNNImputer(missing_values=missing_values, n_neighbors=knn, weights=weights )
			filled = imputer.fit_transform(values)[:, 3].reshape(len(X.coords[x_coords['Y']].values), len(X.coords[x_coords['X']].values), len(X.coords[x_coords['T']].values))
			regridded = filled
			coords = {
				x_coords['T']: [i for i in X.coords[x_coords['T']].values],
				x_coords['X']: [i for i in X.coords[x_coords['X']].values],
				x_coords['Y']: [i for i in X.coords[x_coords['Y']].values]
			}
			data_vars = { xvarname: ([ x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
			return xr.Dataset(data_vars, coords=coords)


def get_nanmask(X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}):
	assert 'X' in x_coords.keys() and 'Y' in x_coords.keys(), '{} X must have XYT dimensions - not shown in x_coords'.format(dt.datetime.now())
	X = standardize_dims(X, x_coords)
	xvarname = [i for i in X.data_vars][0]
	notnans = getattr(X, xvarname)
	if 'M' in x_coords.keys():
		notnans  = notnans.sum(x_coords['M'], skipna=False)
	if 'T' in x_coords.keys():
		notnans = notnans.sum(x_coords['T'], skipna=False)
	notnans  = notnans.transpose(x_coords['Y'], x_coords['X']).values
	notnans[np.isnan(notnans)] = -999
	notnans[notnans > -999] = 1
	notnans[notnans == -999] = 0
	coords = {
		x_coords['X']: X.coords[x_coords['X']].values,
		x_coords['Y']: X.coords[x_coords['Y']].values,
	}
	data_var = {xvarname: ( [ x_coords['Y'], x_coords['X']], notnans)}
	return xr.Dataset(data_var, coords=coords)

def mask_nan(X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, nanmask=None, missing_value=None):
	assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys() and 'M' in x_coords.keys(), '{} X must have XYTM dimensions - not shown in x_coords'.format(dt.datetime.now())
	X = standardize_dims(X, x_coords)
	xvarname = [i for i in X.data_vars][0]

	if missing_value is None:
		missing_value = float(getattr(X, xvarname).mean())

	if nanmask is None:
		nanmask = get_nanmask(X, x_coords)

	if x_coords['T'] not in nanmask.dims:
		nanmask = nanmask.expand_dims({x_coords['T']: [i for i in range(len(X.coords[x_coords['T']].values))]})
	if x_coords['M'] not in nanmask.dims:
		nanmask = nanmask.expand_dims({x_coords['M']: [-999]})
	nanmask_coords = x_coords
	nanmask_coords['M'] = x_coords['M']
	if not are_same_shape(nanmask, X, x_coords=nanmask_coords, y_coords=x_coords):
		nanmask = regrid(nanmask, X, x_coords=nanmask_coords, y_coords=x_coords)
	nanmaskvarname = [i for i in nanmask.data_vars][0]
	xvarname = [i for i in X.data_vars][0]
	concatted =  concat([getattr(nanmask, nanmaskvarname).transpose(nanmask_coords['M'], nanmask_coords['T'], nanmask_coords['Y'], nanmask_coords['X']), getattr(X, xvarname).transpose(x_coords['M'], x_coords['T'], x_coords['Y'], x_coords['X'])], [nanmask_coords, x_coords], x_coords['M'])
	return concatted.where(concatted.isel(M=0) > 0, other = missing_value)#.isel(M=slice(1, None))


def regrid_fill(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_value=-1, verbose=0, fill='mean'):
	x_coords_slice = {'X':x_coords['X'], 'Y':x_coords['Y'], 'T':x_coords['T']}
	if not are_same_shape(X, Y, x_coords=x_coords_slice, y_coords=y_coords):
		X = regrid(X, Y, x_coords=x_coords, y_coords=y_coords, verbose=verbose, missing_value=missing_value, fill=fill)
		if fill == 'kernel':
			if verbose>1:
				print('{} Filling Missing Values ({}) via kernel average'.format(dt.datetime.now(), missing_value))
			Y = kernel_average(Y, y_coords, weighting=2,  missing_value=missing_value, verbose=verbose)
		elif fill == 'mean':
			if verbose>1:
				print('{} Filling Missing Values ({}) via mean across XY for each T'.format(dt.datetime.now(), missing_value))
			Y = Y.fillna(Y.mean(y_coords['X']).mean(y_coords['Y']))
		elif fill == 'knn':
			if verbose>1:
				print('{} Filling Missing Values ({}) via k-nearest neighbors'.format(dt.datetime.now(), missing_value))
			Y = knn(Y, y_coords, verbose=verbose)
		else:
			assert False, '{} fill must be one of [knn, mean, kernel]'.format(dt.datetime.now())
	else:
		if fill == 'kernel':
			if verbose>1:
				print('{} Filling Missing Values ({}) via kernel average'.format(dt.datetime.now(), missing_value))
			X = kernel_average(X, x_coords, weighting=2, missing_value=missing_value, verbose=verbose )
			Y = kernel_average(Y, y_coords, weighting=2,  missing_value=missing_value, verbose=verbose)
		elif fill == 'mean':
			if verbose>1:
				print('{} Filling Missing Values ({}) via mean across XY for each T'.format(dt.datetime.now(), missing_value))
			X = X.fillna(X.mean(x_coords['X']).mean(x_coords['Y']))
			Y = Y.fillna(Y.mean(y_coords['X']).mean(y_coords['Y']))
		elif fill == 'knn':
			if verbose>1:
				print('{} Filling Missing Values ({}) via k-nearest neighbors'.format(dt.datetime.now(), missing_value))
			X = knn(X, x_coords, verbose=verbose)
			Y = knn(Y, y_coords, verbose=verbose)
		else:
			assert False, '{} fill must be one of [knn, mean, kernel]'.format(dt.datetime.now())
	return X, Y
