from .data_validation import *
import numpy as np
import xarray as xr
import datetime as dt

def inner_join(X, Y, dim, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
	assert dim in ['X', 'Y', 'T', 'M'], 'dim must be one of ["X", "Y", "T", "M"]'
	X = standardize_dims(X, x_coords, verbose=verbose)
	Y = standardize_dims(Y, y_coords, verbose=verbose)
	X = X.dropna(x_coords[dim], how="all")
	Y = Y.dropna(y_coords[dim], how="all")

	drop_vals = [val for val in X.coords[x_coords[dim]].values if val not in Y.coords[y_coords[dim]].values ]
	X = X.drop(drop_vals, dim=x_coords[dim])

	drop_vals = [val for val in Y.coords[y_coords[dim]].values if val not in X.coords[x_coords[dim]].values ]
	Y = Y.drop(drop_vals, dim=y_coords[dim])
	return X, Y



def concat(args, coords,  dim, verbose=False):
	dss, coordss, varnames = [], [], []
	for i in range(len(args)):
		x = standardize_dims(args[i], coords[i], verbose=verbose)
		if i > 0:
			for coord in coords[i].keys():
				if coord != dim:
					assert len(x.coords[coords[i][coord]].values) == len(dss[0].coords[coordss[0][coord]].values), 'Mismatched {} dimension'.format(coord)
		dss.append(x)
		coordss.append(coords[i])
		varname =  [ii for ii in x.data_vars][0]
		varnames.append(varname)

	vals = getattr(dss[0], varnames[0]).values
	mdim = [ x for x in dss[0].coords[coordss[0]['M']].values ]
	for i in range(1, len(args)):
		vals = np.vstack((vals, getattr(dss[i], varnames[i]).values))
		for x in dss[i].coords[coordss[i]['M']].values:
			if x not in mdim:
				mdim.append(x)
			else:
				mdim.append(np.random.randint(1000))
	data_vars = {varnames[0]: ([coordss[0]['M'], coordss[0]['T'], coordss[0]['Y'], coordss[0]['X'] ], vals)}
	coords2 = {
		coordss[0]['T']: dss[0].coords[coordss[0]['T']].values,
		coordss[0]['M']: mdim,
		coordss[0]['X']: dss[0].coords[coordss[0]['X']].values,
		coordss[0]['Y']: dss[0].coords[coordss[0]['Y']].values
	}
	return xr.Dataset(data_vars, coords=coords2)
