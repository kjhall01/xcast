import xarray as xr
from pathlib import Path
import numpy as np

# To keep it simple, ALL data objects in XCAST must satisfy the following
#  1) 4 Dimensional, with Latitude, Longitude, Sample, and Feature dimensions.
#  2) Transposed to the order (Lat, Lon, Sample, Feature)
#  3) Coordinates match dimensions - same names, same sizes.
#  4) Of type "Xarray.DataArray"

def to_xss(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""rename dims to labels required for xskillscore"""
	return X.rename({
		x_lat_dim: 'lat',
		x_lon_dim: 'lon',
		x_sample_dim: 'time',
		x_feature_dim: 'member'
	}).transpose('time', 'member', 'lat', 'lon')

def check_transposed(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X is transposed to [Lat, Lon, Sample, Feature] order"""
	assert list(X.dims).index(x_lat_dim) == 0, 'BaseMME.fit requires X to be transposed to LAT x LON x SAMPLE x FEATURE'
	assert list(X.dims).index(x_lon_dim) == 1, 'BaseMME.fit requires X to be transposed to LAT x LON x SAMPLE x FEATURE'
	assert list(X.dims).index(x_sample_dim) == 2, 'BaseMME.fit requires X to be transposed to LAT x LON x SAMPLE x FEATURE'
	assert list(X.dims).index(x_feature_dim) == 3, 'BaseMME.fit requires X to be transposed to LAT x LON x SAMPLE x FEATURE'

def check_dimensions(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X is 4D, with Dimension Names as specified by x_lat_dim, x_lon_dim, x_sample_dim, and x_feature_dim"""
	assert len(X.dims) == 4, 'BaseMME.fit requires X to be 4-Dimensional'
	assert x_lat_dim in X.dims, 'BaseMME.fit requires x_lat_dim to be a dimension on X'
	assert x_lon_dim in X.dims, 'BaseMME.fit requires x_lon_dim to be a dimension on X'
	assert x_sample_dim in X.dims, 'BaseMME.fit requires x_sample_dim to be a dimension on X'
	assert x_feature_dim in X.dims, 'BaseMME.fit requires x_feature_dim to be a dimension on X'

def check_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X has coordinates named as specified by x_lat_dim, x_lon_dim, x_sample_dim, and x_feature_dim"""
	assert x_lat_dim in X.coords.keys(), 'BaseMME.fit requires x_lat_dim to be a coordinate on X'
	assert x_lon_dim in X.coords.keys(), 'BaseMME.fit requires x_lon_dim to be a coordinate on X'
	assert x_sample_dim in X.coords.keys(), 'BaseMME.fit requires x_sample_dim to be a coordinate on X'
	assert x_feature_dim in X.coords.keys(), 'BaseMME.fit requires x_feature_dim to be a coordinate on X'

def check_consistent(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X's Coordinates are the same length as X's Dimensions"""
	assert X.shape[list(X.dims).index(x_lat_dim)] == len(X.coords[x_lat_dim].values), "BaseMME.fit requires X's x_lat_dim coordinate to be the same length as X's x_lat_dim dimension"
	assert X.shape[list(X.dims).index(x_lon_dim)] == len(X.coords[x_lon_dim].values), "BaseMME.fit requires X's x_lon_dim coordinate to be the same length as X's x_lon_dim dimension"
	assert X.shape[list(X.dims).index(x_sample_dim)] == len(X.coords[x_sample_dim].values), "BaseMME.fit requires X's x_sample_dim coordinate to be the same length as X's x_sample_dim dimension"
	assert X.shape[list(X.dims).index(x_feature_dim)] == len(X.coords[x_feature_dim].values), "BaseMME.fit requires X's x_feature_dim coordinate to be the same length as X's x_feature_dim dimension"

def check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X is an Xarray.DataArray"""
	assert type(X) == xr.DataArray, 'BaseMME.fit requires X to be of type "Xarray.DataArray"'

def check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X satisfies all conditions for XCAST"""
	check_dimensions(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_consistent(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	#check_transposed(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)


def open_CsvDataset(filename, delimiter=',', M='M', T='T', tlabels=False, varnames=False, parameter='climate_var'):
	"""opens a .csv file formatted like n_samples x m_features. returns Xarray DataArray with X=0, Y=0, Samples=N, features=M.
	Can include labels for each sample, and labels for each feature. """
	assert Path(filename).absolute().is_file(), 'Cant find {}'.format(Path(filename).absolute())

	with open(str(Path(filename).absolute()), 'r') as f:
		content = f.read()
	content = [line.strip().split(delimiter) for line in content.split('\n') if len(line) > 0 ]

	# check for variable names
	for i in range(len(content[0])):
		if i != 0:
			try:
				x = float(content[0][i])
			except:
				varnames = True

	# check for time labels
	for i in range(len(content)):
		if i != 0:
			try:
				x = float(content[i][0])
			except:
				tlabels = True

	# extract  variable names
	if varnames:
		varnames = content.pop(0)
		if not tlabels:
			varnames = [ str(i.strip()) for i in varnames ]
		else:
			varnames = [str(varnames[i].strip()) for i in range(1, len(varnames))]
	else:
		varnames = [i+1 for i in range(max([len(content[j]) for j in range(len(content))]))]

	# extract time labels
	if tlabels:
		tlabels = []
		for i in range(len(content)):
			tlabels.append(content[i].pop(0))
	else:
		tlabels = [i+1 for i in range(len(content))]

	# check shape of array - pad to len(varnames) with np.nan, and then cut off extra
	for i in range(len(content)):
		while len(content[i]) < len(varnames):
			content[i].append(np.nan)
		content[i] = content[i][:len(varnames)]
		content[i] = [ float(content[i][j]) for j in range(len(content[i]))]

	content = np.asarray(content)
	content = content.reshape((len(tlabels), len(varnames), 1 , 1 ))
	coords = {M: varnames, T: tlabels, 'X':[0], 'Y': [0]}
	data_vars = {parameter: ([T, M, 'X', 'Y'], content)}
	return xr.Dataset(data_vars, coords=coords)

### BASH Utilities
def rmrf(dirn):
	subfiles = [file for file in dirn.glob('*') if file.is_file()]
	subdirs = [diro for diro in dirn.glob('*') if diro.is_dir()]

	for file in subfiles:
		file.unlink()
	for subdir in subdirs:
		try:
			subdir.rmdir()
		except:
			rmrf(subdir)
	dirn.rmdir()



def list_hstack2d(lists):
	d1 = len(lists[0])
	for lst in lists:
		assert len(lst) == d1, 'first dimension must match on all lists'
	main = lists[0]
	for i in range(1, len(lists)):
		for j in range(d1):
			main[j].extend(lists[i][j])
	return main

def list_vstack2d(lists):
	d2 = len(lists[0][0])
	for lst in lists:
		assert len(lst[0]) == d2, 'second dimension must match on all lists'
	main = lists[0]
	for i in range(1, len(lists)):
		main.extend(lists[i])
	return main

def block(lists):
	hstacked = [ list_hstack2d(lists[i]) for i in range(len(lists))]
	return list_vstack2d(hstacked)
