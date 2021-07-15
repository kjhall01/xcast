import xarray as xr
import datetime as dt

def check_same_shape(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}):
	assert len([key for key in x_coords.keys()]) == len([key for key in y_coords.keys()]), '{} Specified X-Coords and Y-Coords different lengths - X: {}, Y: {}'.format(dt.datetime.now(), x_coords, y_coords)

	for coord in x_coords.keys(): #check all specified x dimensions  specified for y
		assert coord in y_coords.keys(), '{} {}-Coord in specified x_coords not found in specified y_coords - {} '.format(dt.datetime.now(), coord, y_coords)

	for coord in y_coords.keys(): #check all specified y dimensions  specified for x
		assert coord in x_coords.keys(), '{} {}-Coord in specified y_coords not found in specified x_coords - {} '.format(dt.datetime.now(), coord, x_coords)
		assert y_coords[coord] in Y.coords, '{} Specified {}-Coord, {}, not found on Y - {}'.format(dt.datetime.now(), coord, y_coords[coord], Y.coords)

	for coord in x_coords.keys(): # check lenght of all coords are the same
		assert len(X.coords[x_coords[coord]].values) == len(Y.coords[y_coords[coord]].values), '{} Mismatched {}-Dimension - {} on X and {} on Y '.format(dt.datetime.now(), coord, len(X.coords[x_coords[coord]].values), len(Y.coords[y_coords[coord]].values))
		assert x_coords[coord] in X.coords, '{} Specified {}-Coord, {}, not found on X - {}'.format(dt.datetime.now(), coord, x_coords[coord], X.coords)

def are_same_shape(X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}):
	try:
		check_same_shape(X, Y, x_coords=x_coords, y_coords=y_coords)
		return True
	except:
		return False

def standardize_dims(X, coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, var='climate_var', verbose=False):
	assert type(X) in [xr.Dataset, xr.DataArray], 'X must be DataSet or DataArray'
	if type(X) is xr.Dataset:
		var = [i for i in X.data_vars][0] # set var equal to the name of the first variable in the dataset
		X = getattr(X, var) #grab the dataarray

	# remove unspecified dimensions by averaging
	for coord in X.dims:
		if coord not in coords.values():
			if verbose>1:
				print('{} Averaging Out Extraneous Dimension "{}" on X'.format(dt.datetime.now(), coord))
			X = X.mean(coord, skipna=True)

	# fill missing specified dimensions with size-1 dimensions & assign coords
	for coord in coords.keys():
		if coords[coord] not in X.coords:
			if verbose>1:
				print('{} Expanding Missing {} Dimension "{}" on {} '.format(dt.datetime.now(), coord, coords[coord], var))
			X = X.expand_dims({'{}'.format(coords[coord]): [0]})

		if coords[coord] not in X.coords:
			if verbose>1:
				print('{} Assigning {} Coordinates for "{}" on {}'.format(dt.datetime.now(), coord, coords[coord], var))
			X = X.assign_coords({'{}'.format(coords[coord]): [i for i in range(X['{}'.format(coords[coord])].shape[0]) ] })

	# standardize longitude values to [0, 360)
	if 'X' in coords.keys():
		new_x = []
		for i in X.coords[coords['X']].values:
			if i < 0:
				new_x.append(360 + i)
			else:
				new_x.append(i)
		X.coords[coords['X']] = new_x

	return xr.Dataset({var: X}, coords=X.coords)
