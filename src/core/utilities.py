import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd

def shape(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	return X.shape[list(X.dims).index(x_lat_dim)], X.shape[list(X.dims).index(x_lon_dim)], X.shape[list(X.dims).index(x_sample_dim)], X.shape[list(X.dims).index(x_feature_dim)]


def guess_coords(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	dims = [x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim]
	dims = [dim for dim in dims if dim is not None]
	dims.extend(list(X.dims))
	dims = list(set(dims))
	common_x = ['LONGITUDE', 'LONG', 'X', 'LON']
	common_y = ['LATITUDE', 'LAT', 'LATI', 'Y']
	common_t = ['T', 'S', 'TIME', 'SAMPLES', 'SAMPLE', 'INITIALIZATION', 'INIT','D', 'DATE', "TARGET", 'YEAR', 'I', 'N']
	common_m = ['M', 'MODE', 'FEATURES', 'F', 'REALIZATION', 'MEMBER', 'Z', 'C', 'CAT', 'NUMBER', 'V', 'VARIABLE', 'VAR', 'P', 'LEVEL']
	ret = {'lat': x_lat_dim, 'lon': x_lon_dim, 'samp': x_sample_dim, 'feat': x_feature_dim}
	while len(dims) > 0:
		dim = dims.pop(0)
		assigned = False

		for x in common_x:
			if x==dim.upper() and ret['lon'] is None and assigned is False:
				ret['lon'] = dim
				assigned=True
		for y in common_y:
			if y == dim.upper() and ret['lat'] is None and assigned is False:
				ret['lat'] = dim
				assigned=True
		for t in common_t:
			if t==dim.upper() and ret['samp'] is None and assigned is False:
				ret['samp'] = dim
				assigned=True
		for m in common_m:
			if m==dim.upper()  and ret['feat'] is None and assigned is False:
				ret['feat'] = dim
				assigned=True

		for x in common_x:
			if (x in dim.upper()) and ret['lon'] is None and assigned is False:
				ret['lon'] = dim
				assigned=True
		for y in common_y:
			if (y in dim.upper() ) and ret['lat'] is None and assigned is False:
				ret['lat'] = dim
				assigned=True
		for t in common_t:
			if (t in dim.upper() ) and ret['samp'] is None and assigned is False:
				ret['samp'] = dim
				assigned=True
		for m in common_m:
			if (m in dim.upper() )  and ret['feat'] is None and assigned is False:
				ret['feat'] = dim
				assigned=True

		for x in common_x:
			if (x in dim.upper() or dim.upper() in x) and ret['lon'] is None and assigned is False:
				ret['lon'] = dim
				assigned=True
		for y in common_y:
			if (y in dim.upper() or dim.upper() in y) and ret['lat'] is None and assigned is False:
				ret['lat'] = dim
				assigned=True
		for t in common_t:
			if (t in dim.upper() or dim.upper() in t) and ret['samp'] is None and assigned is False:
				ret['samp'] = dim
				assigned=True
		for m in common_m:
			if (m in dim.upper() or dim.upper() in m)  and ret['feat'] is None and assigned is False:
				ret['feat'] = dim
				assigned=True

	assert None not in ret.values(), 'Could not detect one or more dimensions: \n  LATITUDE: {lat}\n  LONGITUDE: {lon}\n  SAMPLE: {samp}\n  FEATURE: {feat}\n'.format(**ret)
	return ret['lat'], ret['lon'], ret['samp'], ret['feat']


def guess_coords2(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	assert type(X) == xr.DataArray, 'X must be a data array'
	common_x = ['LONGITUDE', 'LONG', 'X', 'LON']
	common_y = ['LATITUDE', 'LAT', 'LATI', 'Y']
	common_t = ['T', 'S', 'TIME', 'SAMPLES', 'SAMPLE', 'INITIALIZATION', 'INIT', "TARGET"]
	common_m = ['M', 'FEATURES', 'F', 'REALIZATION', 'MEMBER', 'Z', 'C', 'CAT']
	ret = {'lat': x_lat_dim, 'lon': x_lon_dim, 'samp': x_sample_dim, 'feat': x_feature_dim}
	for dim in X.dims:
		for x in common_x:
			if x in dim.upper() and ret['lon'] is None:
				ret['lon'] = dim
		for y in common_y:
			if y in dim.upper() and ret['lat'] is None:
				ret['lat'] = dim
		for t in common_t:
			if t in dim.upper() and ret['samp'] is None:
				ret['samp'] = dim
		for m in common_m:
			if m in dim.upper() and ret['feat'] is None:
				ret['feat'] = dim
	assert None not in ret.values(), 'Could not detect one or more dimensions: \n  LATITUDE: {lat}\n  LONGITUDE: {lon}\n  SAMPLE: {samp}\n  FEATURE: {feat}\n'.format(**ret)
	vals = []
	for val in ret.values():
		if val not in vals:
			vals.append(val)
	assert len(vals) == 4, 'Detection Faild - Duplicated Coordinate: \n  LATITUDE: {lat}\n  LONGITUDE: {lon}\n  SAMPLE: {samp}\n  FEATURE: {feat}\n'.format(**ret)
	return ret['lat'], ret['lon'], ret['samp'], ret['feat']

def guess_coords_view_prob(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	dims = [x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim]
	dims = [dim for dim in dims if dim is not None]
	dims.extend(list(X.dims))
	dims = list(set(dims))
	common_x = ['LONGITUDE', 'LONG', 'X', 'LON']
	common_y = ['LATITUDE', 'LAT', 'LATI', 'Y']
	common_t = ['T', 'S', 'TIME', 'SAMPLES', 'SAMPLE', 'INITIALIZATION', 'INIT','D', 'DATE', "TARGET", 'YEAR', 'I', 'N']
	common_m = ['M', 'MODE', 'FEATURES', 'F', 'REALIZATION', 'MEMBER', 'Z', 'C', 'CAT', 'NUMBER', 'V', 'VARIABLE', 'VAR', 'P', 'LEVEL']
	ret = {'lat': x_lat_dim, 'lon': x_lon_dim, 'samp': x_sample_dim, 'feat': x_feature_dim}
	while len(dims) > 0:
		dim = dims.pop(0)
		assigned = False

		for x in common_x:
			if x==dim.upper() and ret['lon'] is None and assigned is False:
				ret['lon'] = dim
				assigned=True
		for y in common_y:
			if y == dim.upper() and ret['lat'] is None and assigned is False:
				ret['lat'] = dim
				assigned=True
		for t in common_t:
			if t==dim.upper() and ret['samp'] is None and assigned is False:
				ret['samp'] = dim
				assigned=True
		for m in common_m:
			if m==dim.upper()  and ret['feat'] is None and assigned is False:
				ret['feat'] = dim
				assigned=True

		for x in common_x:
			if (x in dim.upper()) and ret['lon'] is None and assigned is False:
				ret['lon'] = dim
				assigned=True
		for y in common_y:
			if (y in dim.upper() ) and ret['lat'] is None and assigned is False:
				ret['lat'] = dim
				assigned=True
		for t in common_t:
			if (t in dim.upper() ) and ret['samp'] is None and assigned is False:
				ret['samp'] = dim
				assigned=True
		for m in common_m:
			if (m in dim.upper() )  and ret['feat'] is None and assigned is False:
				ret['feat'] = dim
				assigned=True

		for x in common_x:
			if (x in dim.upper() or dim.upper() in x) and ret['lon'] is None and assigned is False:
				ret['lon'] = dim
				assigned=True
		for y in common_y:
			if (y in dim.upper() or dim.upper() in y) and ret['lat'] is None and assigned is False:
				ret['lat'] = dim
				assigned=True
		for t in common_t:
			if (t in dim.upper() or dim.upper() in t) and ret['samp'] is None and assigned is False:
				ret['samp'] = dim
				assigned=True
		for m in common_m:
			if (m in dim.upper() or dim.upper() in m)  and ret['feat'] is None and assigned is False:
				ret['feat'] = dim
				assigned=True
	return ret['lat'], ret['lon'], ret['samp'], ret['feat']


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
	assert list(X.dims).index(x_lat_dim) == 0, 'XCast requires a dataset to be transposed to LAT x LON x SAMPLE x FEATURE'
	assert list(X.dims).index(x_lon_dim) == 1, 'XCast requires a dataset to be transposed to LAT x LON x SAMPLE x FEATURE'
	assert list(X.dims).index(x_sample_dim) == 2, 'XCast requires a dataset to be transposed to LAT x LON x SAMPLE x FEATURE'
	assert list(X.dims).index(x_feature_dim) == 3, 'XCast requires a dataset to be transposed to LAT x LON x SAMPLE x FEATURE'

def check_dimensions(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X is 4D, with Dimension Names as specified by x_lat_dim, x_lon_dim, x_sample_dim, and x_feature_dim"""
	assert 4 <= len(X.dims) <= 5, 'XCast requires a dataset to be 4-Dimensional'
	assert x_lat_dim in X.dims, 'XCast requires a dataset_lat_dim to be a dimension on X'
	assert x_lon_dim in X.dims, 'XCast requires a dataset_lon_dim to be a dimension on X'
	assert x_sample_dim in X.dims, 'XCast requires a dataset_sample_dim to be a dimension on X'
	assert x_feature_dim in X.dims, 'XCast requires a dataset_feature_dim to be a dimension on X'

def check_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X has coordinates named as specified by x_lat_dim, x_lon_dim, x_sample_dim, and x_feature_dim"""
	assert x_lat_dim in X.coords.keys(), 'XCast requires a dataset_lat_dim to be a coordinate on X'
	assert x_lon_dim in X.coords.keys(), 'XCast requires a dataset_lon_dim to be a coordinate on X'
	assert x_sample_dim in X.coords.keys(), 'XCast requires a dataset_sample_dim to be a coordinate on X'
	assert x_feature_dim in X.coords.keys(), 'XCast requires a dataset_feature_dim to be a coordinate on X'

def check_consistent(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X's Coordinates are the same length as X's Dimensions"""
	assert X.shape[list(X.dims).index(x_lat_dim)] == len(X.coords[x_lat_dim].values), "XCast requires a dataset's x_lat_dim coordinate to be the same length as its x_lat_dim dimension"
	assert X.shape[list(X.dims).index(x_lon_dim)] == len(X.coords[x_lon_dim].values), "XCast requires a dataset's x_lon_dim coordinate to be the same length as its x_lon_dim dimension"
	assert X.shape[list(X.dims).index(x_sample_dim)] == len(X.coords[x_sample_dim].values), "XCast requires a dataset's x_sample_dim coordinate to be the same length as its x_sample_dim dimension"
	assert X.shape[list(X.dims).index(x_feature_dim)] == len(X.coords[x_feature_dim].values), "XCast requires a dataset's x_feature_dim coordinate to be the same length as its x_feature_dim dimension"

def check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X is an Xarray.DataArray"""
	assert type(X) == xr.DataArray, 'XCast requires a dataset to be of type "Xarray.DataArray"'

def check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim):
	"""Checks that X satisfies all conditions for XCAST"""
	check_dimensions(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_consistent(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	#check_transposed(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

def check_xyt_compatibility(X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None):
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	xlat, xlon, xsamp, xfeat = shape(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
	ylat, ylon, ysamp, yfeat = shape(Y, x_lat_dim=y_lat_dim, x_lon_dim=y_lon_dim, x_sample_dim=y_sample_dim, x_feature_dim=y_feature_dim)

	assert xlat == ylat, "XCAST model training requires X and Y to have the same dimensions across XYT - latitude mismatch"
	assert xlon == ylon, "XCAST model training requires X and Y to have the same dimensions across XYT - longitude mismatch"
	assert xsamp == ysamp, "XCAST model training requires X and Y to have the same dimensions across XYT - sample mismatch"

def open_csvdataset(filename, delimiter='\t', name='variable'):
	"""wrapper for pd.read_csv which gets you an xarray dataarray"""
	assert Path(filename).absolute().is_file(), 'Cant find {}'.format(Path(filename).absolute())
	df = pd.read_csv(filename, delimiter=delimiter)
	return xr.DataArray(name=name, data=df[df.columns[1:]].values, dims=[df.columns[0], 'M'], coords={df.columns[0]: df[df.columns[0]].values, 'M': df.columns[1:].values} )


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
