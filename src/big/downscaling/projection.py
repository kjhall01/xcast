from ..core import *
from scipy.interpolate import interp2d
import numpy as np
from sklearn.impute import KNNImputer
import datetime as dt
import uuid
from pathlib import Path
from dask import delayed, compute
import sys

def regrid(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_value=-999, interp='linear', fill='mean', verbose=False, x_chunks=10, y_chunks=10, t_chunks=100):
	tf_uuid = str(uuid.uuid4())
	x_ids, y_ids = [], []
	jobs = []
	if 'M' in x_coords.keys():
		if 'M' not in y_coords.keys():
			y_coords['M'] = x_coords['M'] # coerce Y onto X's M dim
		X = standardize_dims(X, x_coords , verbose=verbose)
		Y = standardize_dims(Y, y_coords, verbose=verbose)

		chunksx  = {
			x_coords['T']: max(1,int( X.dims[x_coords['T']] / t_chunks)),
			x_coords['M']: max(1, int( X.dims[x_coords['M']])),
			x_coords['X']: max(1, int( X.dims[x_coords['X']])),
			x_coords['Y']: max(1, int( X.dims[x_coords['Y']] ))
		}
		chunksy  = {
			y_coords['T']: max(1,int(Y.dims[y_coords['T']] / t_chunks)),
			y_coords['M']: max(1, int(Y.dims[y_coords['M']])),
			y_coords['X']: max(1,int(Y.dims[y_coords['X']])),
			y_coords['Y']: max(int(Y.dims[y_coords['Y']] ), 1)
		}
		X1 = X.transpose(x_coords['T'], x_coords['M'], x_coords['Y'], x_coords['X']).chunk(chunksx)
		Y1 = Y.transpose(y_coords['T'], y_coords['M'], y_coords['Y'], y_coords['X']).chunk(chunksy)

		if fill == 'kernel':
			X1 = kernel_average(X1, x_coords, weighting=2, verbose=verbose , missing_value = missing_value)
			Y1 = kernel_average(Y1, y_coords, weighting=2, verbose=verbose, missing_value = missing_value)
		elif fill == 'mean':
			X1 = X.fillna(X1.mean(x_coords['X']).mean(x_coords['Y']))
			Y1 = Y1.fillna(Y1.mean(y_coords['X']).mean(y_coords['Y']))
		elif fill == 'knn':
			X1 = knn(X1, x_coords, verbose=verbose)
			Y1 = knn(Y1, y_coords, verbose=verbose)
		else:
			assert False, '{} fill must be one of [knn, mean, kernel]'.format(dt.datetime.now())

		count=0
		if verbose > 1:
			total = len(X1.chunks[x_coords['T']]) * len(X1.chunks[x_coords['M']])
			print('{} Regridding X: ['.format(dt.datetime.now() ) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		for i in range(len(X1.chunks[x_coords['T']])): #chunks in T
			indx = 0
			for ii in range(i):
				indx += X1.chunks[x_coords['T']][ii]
			for j in range(len(X1.chunks[x_coords['M']])):
				jndx = 0
				for jj in range(j):
					jndx += X1.chunks[x_coords['M']][jj]
				iseldict = {x_coords['M']: slice(jndx, jndx+X1.chunks[x_coords['M']][j]), x_coords['T']: slice(indx, indx+X1.chunks[x_coords['T']][i])  }
				iseldicty = {y_coords['M']: slice(jndx, jndx+Y1.chunks[y_coords['M']][0]), y_coords['T']: slice(indx, indx+Y1.chunks[y_coords['T']][i])  }
				X = regrid_one(X1.isel(**iseldict), Y1.isel(**iseldicty), x_coords=x_coords, y_coords=y_coords,  verbose=verbose, fill=fill)
				x_ids.append(Path().home() / '.xcast_cache' / 'X_{}_{}_{}'.format(tf_uuid, i , j))
				count += 1
				if  verbose > 1:
					print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
					sys.stdout.flush()
				X.to_netcdf(Path().home() / '.xcast_cache' / 'X_{}_{}_{}'.format(tf_uuid, i , j))
				#Y.to_netcdf(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}'.format(tf_uuid, i , j))
				#y_ids.append(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}'.format(tf_uuid, i , j))

		#compute(*jobs)
		if verbose > 1:
			print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*25 +'] 100% ({}/{})'.format( total, total))
	else:
		if 'M' in y_coords.keys():
			del y_coords['M']  # coerce Y onto X's M dim
		X = standardize_dims(X, x_coords , verbose=verbose)
		Y = standardize_dims(Y, y_coords, verbose=verbose)

		chunksx  = {
			x_coords['T']: max(1,int( X.dims[x_coords['T']] / t_chunks)),
			x_coords['X']: max(1, int( X.dims[x_coords['X']] )),
			x_coords['Y']: max(1, int( X.dims[x_coords['Y']] ))
		}
		chunksy  = {
			y_coords['T']: max(1,int(len(Y.coords[y_coords['T']].values))),
			y_coords['X']: max(1,int(len(Y.coords[y_coords['X']].values) )),
			y_coords['Y']: max(int(len(Y.coords[y_coords['Y']].values) ), 1)
		}
		X1 = X.transpose(x_coords['T'],  x_coords['Y'], x_coords['X']).chunk(chunksx)
		Y1 = Y.transpose(y_coords['T'], y_coords['Y'], y_coords['X']).chunk(chunksy)

		if fill == 'kernel':
			X1 = kernel_average(X1, x_coords, weighting=2, verbose=verbose , missing_value = missing_value)
			Y1 = kernel_average(Y1, y_coords, weighting=2, verbose=verbose, missing_value = missing_value)
		elif fill == 'mean':
			X1 = X.fillna(X1.mean(x_coords['X']).mean(x_coords['Y']))
			Y1 = Y1.fillna(Y1.mean(y_coords['X']).mean(y_coords['Y']))
		elif fill == 'knn':
			X1 = knn(X1, x_coords, verbose=verbose)
			Y1 = knn(Y1, y_coords, verbose=verbose)
		else:
			assert False, '{} fill must be one of [knn, mean, kernel]'.format(dt.datetime.now())

		count=0
		if verbose > 1:
			total = len(X1.chunks[x_coords['T']]) * len(X.chunks[x_coords['M']])
			print('{} Regridding X: ['.format(dt.datetime.now() ) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')


		for i in range(len(X1.chunks[x_coords['T']])): #chunks in T
			indx = 0
			for ii in range(i):
				indx += X1.chunks[x_coords['T']][ii]
			iseldict = {x_coords['T']: slice(indx, indx+X1.chunks[x_coords['T']][i])  }
			iseldicty = {y_coords['T']: slice(indx, indx+Y1.chunks[y_coords['T']][i])  }
			X2 = regrid_one(X1.isel(**iseldict), Y1.isel(**iseldicty), x_coords=x_coords, y_coords=y_coords,  verbose=verbose, fill=fill)
			x_ids.append(Path().home() / '.xcast_cache' / 'X_{}_{}_{}'.format(tf_uuid, i , j))
			count += 1
			if  verbose > 1:
				print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
				sys.stdout.flush()
			X2.to_netcdf(Path().home() / '.xcast_cache' / 'X_{}_{}_{}'.format(tf_uuid, i , j))

			#Y.to_netcdf(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}'.format(tf_uuid, i , j))
			#y_ids.append(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}'.format(tf_uuid, i , j))

		if verbose > 1:
			print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*25 +'] 100% ({}/{})'.format( total, total))
	return xr.open_mfdataset(x_ids, chunks='auto', parallel=True)


def regrid_one(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_value=-999, interp='linear', fill='mean', verbose=False):
	X = standardize_dims(X, x_coords , verbose=verbose)
	Y = standardize_dims(Y, y_coords, verbose=verbose)
	xvarname = [i for i in X.data_vars][0]
	check_same_shape(X, Y, {'T':x_coords['T']}, {'T':y_coords['T']})
	if are_same_shape(X, Y, {'X':x_coords['X'], 'Y':x_coords['Y']}, {'X':y_coords['X'], 'Y':y_coords['Y']}):
		return X

	xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]

	x_data = getattr(X, xvarname).values
	#print(X2)
	#print(x_data.shape)
	y_data = getattr(Y, yvarname).values
	#print(Y)
	#print(y_data.shape)
	x_vals, y_vals = X.coords[x_coords['X']].values, X.coords[x_coords['Y']].values
	x_vals2, y_vals2 = Y.coords[y_coords['X']].values, Y.coords[y_coords['Y']].values

	if 'M' in x_coords.keys():
		regridded = []
		for i in range(x_data.shape[0]):
			regridded.append([])
			for j in range(x_data.shape[1]):
				#if j == 0:
				#iseldict = {x_coords['T']: j, x_coords['M']: i}
				#vals = getattr(X2, xvarname).isel(**iseldict)
				interp_func  = interp2d(x_vals, y_vals, x_data[i, j, :, :], kind=interp)

				#iseldict = {y_coords['T']: j, y_coords['M']: 0}
				#vals2 = getattr(Y, yvarname).isel(**iseldict)
				#x_vals, y_vals = vals2.coords[y_coords['X']].values, vals2.coords[y_coords['Y']].values
				interped = interp_func(x_vals2, y_vals2)
				regridded[i].append(interped )
		regridded = np.asarray(regridded)
		coords = {
			x_coords['T']: [i for i in X.coords[x_coords['T']].values],
			x_coords['M']: [i for i in X.coords[x_coords['M']].values],
			x_coords['X']: [i for i in Y.coords[y_coords['X']].values],
			x_coords['Y']: [i for i in Y.coords[y_coords['Y']].values]
		}
		data_vars = { xvarname: ([x_coords['M'], x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
		return xr.Dataset(data_vars, coords=coords)
	else:
		regridded = []
		for j in range( X.dims[x_coords['T']]):
			#if j == 0:
			#iseldict = {x_coords['T']: j}
			#vals = getattr(X2, xvarname).isel(**iseldict)
			#x_vals, y_vals = vals.coords[x_coords['X']].values, vals.coords[x_coords['Y']].values
			interp_func  = interp2d(x_vals, y_vals, x_data[j, :, :], kind=interp)

			#iseldict = {y_coords['T']: j}
			#vals2 = getattr(Y, yvarname).isel(**iseldict)
			#x_vals, y_vals = vals2.coords[y_coords['X']].values, vals2.coords[y_coords['Y']].values
			interped = interp_func(x_vals2, y_vals2)
			regridded.append(interped )
		regridded = np.asarray(regridded)
		coords = {
			x_coords['T']: [i for i in X.coords[x_coords['T']].values],
			x_coords['X']: [i for i in Y.coords[y_coords['X']].values],
			x_coords['Y']: [i for i in Y.coords[y_coords['Y']].values]
		}
		data_vars = { xvarname: ([ x_coords['T'], x_coords['Y'], x_coords['X']], regridded)}
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
		for i in range( X.dims[x_coords['M']]):
			regridded.append([])
			for j in range( X.dims[x_coords['T']]):
				iseldict = {x_coords['T']: j, x_coords['M']: i}
				vals = getattr(X, xvarname).isel(**iseldict).values
				nmissing = np.sum(np.isnan(vals))
				if nmissing == vals.shape[0] * vals.shape[1]:
					if verbose > 2:
						print( '{} cannot kernel average a fully missing slice'.format(dt.datetime.now()))
					regridded[i].append(vals)
				else:
					while nmissing > 0:
						vals = _kernel_average(vals, weighting=weighting, missing_value=missing_value)
						nmissing = np.sum(np.isnan(vals))
					regridded[i].append( vals )
					count += 1
					if count % 10 == 1 and verbose > 1:
						pct = int( 25* count / ( X.dims[x_coords['M']] *  X.dims[x_coords['T']]))
						print('{} Kernel Average: ['.format(dt.datetime.now()) +  '*'*pct + ' '*(25 - pct) + '] {}% ({}/{})'.format(int(count / ( X.dims[x_coords['M']] *  X.dims[x_coords['T']])*100), count, ( X.dims[x_coords['M']] *  X.dims[x_coords['T']])), end='\r')
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
		for j in range( X.dims[x_coords['T']]):
			iseldict = {x_coords['T']: j}
			vals = getattr(X, xvarname).isel(**iseldict).values
			nmissing = np.sum(np.isnan(vals))
			if nmissing == vals.shape[0] * vals.shape[1]:
				if verbose > 2:
					print( '{} cannot kernel average a fully missing slice'.format(dt.datetime.now()))
				regridded[i].append(vals)
			else:
				while nmissing > 0:
					vals = _kernel_average(vals, weighting=weighting, missing_value=missing_value)
					nmissing = np.sum(np.isnan(vals))
			regridded.append( vals )
			count += 1
			if count % 10 == 1 and verbose > 1:
				pct = int( 25* count / ( X.dims[x_coords['T']]))
				print('{} Kernel Average: ['.format(dt.datetime.now()) +  '*'*pct + ' '*(25 - pct) + '] {}% ({}/{})'.format(int(count / ( X.dims[x_coords['T']])*100), count, (  X.dims[x_coords['T']])), end='\r')
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
			for i in range( X.dims[x_coords['M']]):
				regridded.append([])
				for j in range( X.dims[x_coords['T']]):
					iseldict = {x_coords['T']: j, x_coords['M']: i}
					vals = getattr(X, xvarname).isel(**iseldict).stack(z=(x_coords['X'], x_coords['Y']))
					xvals = np.asarray([n[0] for n in vals.coords['z'].values]).reshape(-1, 1)
					yvals = np.asarray([n[1] for n in vals.coords['z'].values]).reshape(-1, 1)
					c = np.hstack((xvals, yvals))
					values = np.hstack((c, vals.values.reshape(-1,1)))
					imputer = KNNImputer(missing_values=missing_values, n_neighbors=knn, weights=weights )
					filled = imputer.fit_transform(values)[:, 2].reshape( X.dims[x_coords['Y']],  X.dims[x_coords['X']])
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
			for j in range( X.dims[x_coords['T']]):
				iseldict = {x_coords['T']: j}
				vals = getattr(X, xvarname).isel(**iseldict).stack(z=(x_coords['X'], x_coords['Y']))
				xvals = np.asarray([n[0] for n in vals.coords['z'].values]).reshape(-1, 1)
				yvals = np.asarray([n[1] for n in vals.coords['z'].values]).reshape(-1, 1)
				c = np.hstack((xvals, yvals))
				values = np.hstack((c, vals.values.reshape(-1,1)))
				imputer = KNNImputer(missing_values=missing_values, n_neighbors=knn, weights=weights )
				filled = imputer.fit_transform(values)[:, 2].reshape( X.dims[x_coords['Y']],  X.dims[x_coords['X']])
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
			for i in range( X.dims[x_coords['M']]):
				iseldict = { x_coords['M']: i}
				vals = getattr(X, xvarname).isel(**iseldict).stack(z=(x_coords['X'], x_coords['Y'], x_coords['T']))
				xvals = np.asarray([n[0] for n in vals.coords['z'].values]).reshape(-1, 1)
				yvals = np.asarray([n[1] for n in vals.coords['z'].values]).reshape(-1, 1)
				tvals = np.asarray([ii for ii in range(len(vals.coords['z'].values))]).reshape(-1, 1)
				c = np.hstack((xvals, yvals, tvals))
				values = np.hstack((c, vals.values.reshape(-1,1)))
				imputer = KNNImputer(missing_values=missing_values, n_neighbors=knn, weights=weights )
				filled = imputer.fit_transform(values)[:, 3].reshape( X.dims[x_coords['Y']],  X.dims[x_coords['X']],  X.dims[x_coords['T']])
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
			filled = imputer.fit_transform(values)[:, 3].reshape( X.dims[x_coords['Y']],  X.dims[x_coords['X']],  X.dims[x_coords['T']])
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
		nanmask = nanmask.expand_dims({x_coords['T']: [i for i in range( X.dims[x_coords['T']])]})
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


def regrid_fill(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_value=-1, verbose=0, fill='mean', x_chunks=10, y_chunks=10, t_chunks=200):
	tf_uuid = str(uuid.uuid4())
	x_ids, y_ids = [], []
	x_coords_slice = { 'T':x_coords['T']}
	y_coords_slice = { 'T':y_coords['T']}
	check_same_shape(X, Y, x_coords_slice, y_coords_slice)

	x_coords_slice = { 'T':x_coords['T'], 'X':x_coords['X'], 'Y': x_coords['Y']}
	y_coords_slice = { 'T':y_coords['T'], 'X':y_coords['X'], 'Y': y_coords['Y']}
	if are_same_shape(X, Y, x_coords, y_coords):
		return X, Y
	if 'M' in x_coords.keys():
		if 'M' not in y_coords.keys():
			y_coords['M'] = x_coords['M'] # coerce Y onto X's M dim
		X = standardize_dims(X, x_coords , verbose=verbose)
		Y = standardize_dims(Y, y_coords, verbose=verbose)


		chunksx  = {
			x_coords['T']:  max(1, int( X.dims[x_coords['T']] / t_chunks)),
			x_coords['M']:  int( X.dims[x_coords['M']]),
			x_coords['X']: max(1, int( X.dims[x_coords['X']] / x_chunks) ),
			x_coords['Y']: max(1, int( X.dims[x_coords['Y']] / y_chunks)),
		}
		chunksy  = {
			y_coords['T']: max(1, int(len(Y.coords[y_coords['T']].values) / t_chunks)),
			y_coords['M']:  int(len(Y.coords[y_coords['M']].values)),
			y_coords['X']: max(1, int(len(Y.coords[y_coords['X']].values) / x_chunks)),
			y_coords['Y']: max(1, int(len(Y.coords[y_coords['Y']].values) / y_chunks )),
		}
		#if verbose > 2:
		#	print('{} Rechunking X'.format(dt.datetime.now()))
		X1 = X.chunk(chunksx)
		#if verbose > 2:
		#	print('{} Rechunking Y'.format(dt.datetime.now()))
		Y1 = Y.chunk(chunksy)
		count=0
		if verbose > 1:
			total = t_chunks
			print('{} Regridding X: ['.format(dt.datetime.now() ) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')


		#for i in range(len(X1.chunks[x_coords['T']])): #chunks in Y
		#	indx = 0
		#	for ii in range(i):
		#		indx += X1.chunks[x_coords['T']][ii]
		#	iseldict = { x_coords['T']: slice(indx, indx+X1.chunks[x_coords['T']][i])  }
		#	iseldicty = { y_coords['T']: slice(indx, indx+Y1.chunks[y_coords['T']][i])  }
		for i in range(0, len(X1.coords[x_coords['T']].values), chunksx[x_coords['T']]):
			iseldict = { x_coords['T']: slice(i, i+chunksx[x_coords['T']])  }
			iseldicty = { y_coords['T']: slice(i, i+chunksx[x_coords['T']])  }
			X2, Y2 = regrid_fill_one(X1.isel(**iseldict), Y1.isel(**iseldicty), x_coords=x_coords, y_coords=y_coords,  verbose=verbose)
			X2.to_netcdf(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, i, i ))
			Y2.to_netcdf(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, i , i))
			del X2
			del Y2
			x_ids.append(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, i , i))
			y_ids.append(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, i , i))
			count += 1
			if  verbose > 1:
				print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
		if len(X1.coords[x_coords['T']].values) % chunksx[x_coords['T']] > 0:
			iseldict = { x_coords['T']: slice(len(X1.coords[x_coords['T']].values) - (len(X1.coords[x_coords['T']].values) % chunksx[x_coords['T']]), None)  }
			iseldicty = { y_coords['T']: slice(len(X1.coords[x_coords['T']].values) - (len(X1.coords[x_coords['T']].values) % chunksx[x_coords['T']]), None)  }
			X2, Y2 = regrid_fill_one(X1.isel(**iseldict), Y1.isel(**iseldicty), x_coords=x_coords, y_coords=y_coords,  verbose=verbose)
			X2.to_netcdf(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, 'last', 'last' ))
			Y2.to_netcdf(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, 'last' , 'last'))
			del X2
			del Y2
			x_ids.append(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, 'last' , 'last'))
			y_ids.append(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, 'last' , 'last'))
		if verbose > 1:
			print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*25 +'] 100% ({}/{})'.format( total, total))
	else:
		if 'M' in y_coords.keys():
			del y_coords['M'] # coerce Y onto X's M dim
		X = standardize_dims(X, x_coords , verbose=verbose)
		Y = standardize_dims(Y, y_coords, verbose=verbose)

		chunksx  = {
			x_coords['T']:  max(1, int( X.dims[x_coords['T']] / t_chunks)),
			x_coords['X']: max(1, int( X.dims[x_coords['X']] / x_chunks) ),
			x_coords['Y']: max(1, int( X.dims[x_coords['Y']] / y_chunks)),
		}
		chunksy  = {
			y_coords['T']: max(1, int(len(Y.coords[y_coords['T']].values) / t_chunks)),
			y_coords['X']: max(1, int(len(Y.coords[y_coords['X']].values) / x_chunks)),
			y_coords['Y']: max(1, int(len(Y.coords[y_coords['Y']].values) / y_chunks )),
		}

		#if verbose > 2:
		#	print('{} Rechunking X'.format(dt.datetime.now()))
		X1 = X.chunk(chunksx)
		#if verbose > 2:
		#	print('{} Rechunking Y'.format(dt.datetime.now()))
		Y1 = Y.chunk(chunksy)

		count=0
		if verbose > 1:
			total = t_chunks
			print('{} Regridding X: ['.format(dt.datetime.now() ) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

		#for i in range(len(X1.chunks[x_coords['T']])): #chunks in Y
		#	indx = 0
		#	for ii in range(i):
		#		indx += X1.chunks[x_coords['T']][ii]
		#	print(indx, indx+X1.chunks[x_coords['T']][i])
		#	iseldict = { x_coords['T']: slice(indx, indx+X1.chunks[x_coords['T']][i])  }
		#	iseldicty = { y_coords['T']: slice(indx, indx+Y1.chunks[y_coords['T']][i])  }
		for i in range(0, len(X1.coords[x_coords['T']].values), chunksx[x_coords['T']]):
			iseldict = { x_coords['T']: slice(i, i+chunksx[x_coords['T']])  }
			iseldicty = { y_coords['T']: slice(i, i+chunksx[x_coords['T']])  }

			X2, Y2 = regrid_fill_one(X1.isel(**iseldict), Y1.isel(**iseldicty), x_coords=x_coords, y_coords=y_coords,  verbose=verbose)
			X2.to_netcdf(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, i, i ))
			Y2.to_netcdf(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, i , i))
			del X2
			del Y2
			x_ids.append(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, i , i))
			y_ids.append(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, i , i))
			count += 1
			if  verbose > 1:
				print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')
		if len(X1.coords[x_coords['T']].values) % chunksx[x_coords['T']] > 0:
			iseldict = { x_coords['T']: slice(len(X1.coords[x_coords['T']].values) - (len(X1.coords[x_coords['T']].values) % chunksx[x_coords['T']]), None)  }
			iseldicty = { y_coords['T']: slice(len(X1.coords[x_coords['T']].values) - (len(X1.coords[x_coords['T']].values) % chunksx[x_coords['T']]), None)  }

			X2, Y2 = regrid_fill_one(X1.isel(**iseldict), Y1.isel(**iseldicty), x_coords=x_coords, y_coords=y_coords,  verbose=verbose)
			X2.to_netcdf(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, 'last', 'last' ))
			Y2.to_netcdf(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, 'last' , 'last'))
			del X2
			del Y2
			x_ids.append(Path().home() / '.xcast_cache' / 'X_{}_{}_{}.nc'.format(tf_uuid, 'last' , 'last'))
			y_ids.append(Path().home() / '.xcast_cache' / 'Y_{}_{}_{}.nc'.format(tf_uuid, 'last' , 'last'))

		if verbose > 1:
			print('{} Regridding X: ['.format(dt.datetime.now()) + '*'*25 +'] 100% ({}/{})'.format( total, total))
	return xr.open_mfdataset(x_ids, chunks=chunksx, parallel=True), xr.open_mfdataset(y_ids, chunks=chunksy, parallel=True)


def regrid_fill_one(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, missing_value=-1, verbose=0, fill='mean'):
	x_coords_slice = {'X':x_coords['X'], 'Y':x_coords['Y'], 'T':x_coords['T']}
	if not are_same_shape(X, Y, x_coords=x_coords_slice, y_coords=y_coords):
		X1 = regrid_one(X, Y, x_coords=x_coords, y_coords=y_coords, verbose=verbose, missing_value=missing_value, fill=fill)
		if fill == 'kernel':
			if verbose>1:
				print('{} Filling Missing Values ({}) via kernel average'.format(dt.datetime.now(), missing_value))
			Y1 = kernel_average(Y, y_coords, weighting=2,  missing_value=missing_value, verbose=verbose)
		elif fill == 'mean':
			if verbose>3:
				print('{} Filling Missing Values ({}) via mean across XY for each T'.format(dt.datetime.now(), missing_value))
			Y1 = Y.fillna(Y.mean(y_coords['X']).mean(y_coords['Y']))
		elif fill == 'knn':
			if verbose>3:
				print('{} Filling Missing Values ({}) via k-nearest neighbors'.format(dt.datetime.now(), missing_value))
			Y1 = knn(Y, y_coords, verbose=verbose)
		else:
			assert False, '{} fill must be one of [knn, mean, kernel]'.format(dt.datetime.now())
	else:
		if fill == 'kernel':
			if verbose>3:
				print('{} Filling Missing Values ({}) via kernel average'.format(dt.datetime.now(), missing_value))
			X1 = kernel_average(X, x_coords, weighting=2, missing_value=missing_value, verbose=verbose )
			Y1 = kernel_average(Y, y_coords, weighting=2,  missing_value=missing_value, verbose=verbose)
		elif fill == 'mean':
			if verbose>3:
				print('{} Filling Missing Values ({}) via mean across XY for each T'.format(dt.datetime.now(), missing_value))
			X1 = X.fillna(X.mean(x_coords['X']).mean(x_coords['Y']))
			Y1 = Y.fillna(Y.mean(y_coords['X']).mean(y_coords['Y']))
		elif fill == 'knn':
			if verbose>3:
				print('{} Filling Missing Values ({}) via k-nearest neighbors'.format(dt.datetime.now(), missing_value))
			X1 = knn(X, x_coords, verbose=verbose)
			Y1 = knn(Y, y_coords, verbose=verbose)
		else:
			assert False, '{} fill must be one of [knn, mean, kernel]'.format(dt.datetime.now())
	return X1, Y1
