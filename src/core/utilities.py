import xarray as xr
import numpy as np
import pandas as pd

def shape(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	return X.shape[list(X.dims).index(x_lat_dim)], X.shape[list(X.dims).index(x_lon_dim)], X.shape[list(X.dims).index(x_sample_dim)], X.shape[list(X.dims).index(x_feature_dim)]


often_used = { 
	'longitude': ['LONGITUDE', 'LONG', 'X', 'LON'],
	'latitude': ['LATITUDE', 'LAT', 'LATI', 'Y'],
	'sample': ['T', 'S', 'TIME', 'SAMPLES', 'SAMPLE', 'INITIALIZATION', 'INIT','D', 'DATE', "TARGET", 'YEAR', 'I', 'N'],
	'feature': ['M', 'MODE', 'FEATURES', 'F', 'REALIZATION', 'MEMBER', 'Z', 'C', 'CAT', 'NUMBER', 'V', 'VARIABLE', 'VAR', 'P', 'LEVEL'],
}

def guess_coords(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
	ret = {'latitude': x_lat_dim, 'longitude': x_lon_dim, 'sample': x_sample_dim, 'feature': x_feature_dim}
	user_provided_labels = [ ret[i] for i in ret.keys() if ret[i] is not None]
	labels_on_x = list(X.dims)
	for label in user_provided_labels: 
		assert label in labels_on_x, 'user-provided dimension ({}) not found on data-array: {}'.format(label, labels_on_x)
	labels_on_x_minus_user_provided = [ label for label in labels_on_x if label not in user_provided_labels]
	dims_left_to_find = [i for i in ret.keys() if ret[i] is None]
	for label in labels_on_x_minus_user_provided:
		for dim in dims_left_to_find: 
			for candidate in often_used[dim][::-1]:
				if label.upper() == candidate: 
					ret[dim] = label
					dims_left_to_find.pop(dims_left_to_find.index(dim))
	assigned_labels = [ ret[i] for i in ret.keys() if ret[i] is not None]
	unassigned_labels = [ label for label in labels_on_x if label not in assigned_labels]
	if len(unassigned_labels) == 1 and len(dims_left_to_find) == 1: 
		ret[dims_left_to_find[0]] = unassigned_labels[0]
		unassign_labels.pop(0)
		dims_left_to_find.pop(0)

	if len(unassigned_labels) > 0: 
		print('UNABLE TO ASSIGN FOLLOWING LABELS: {}'.format(unassigned_labels))
	if len(dims_left_to_find) > 0: 
		print('UNABLE TO FIND NAMES FOR FOLLOWING DIMS: {}'.format(dims_left_to_find))
	return ret['latitude'], ret['longitude'], ret['sample'], ret['feature']


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



import xarray as xr 
import json
import numpy as np 

def load_parameters(file='test.params'): 
    lats, lons = [], []
    with open(file, 'r') as f: 
        for line in f: 
            i, j = line.split(' ')[:2]
            lats.append(i)
            lons.append(j)
    lats = np.asarray(list(set(lats)))
    lons = np.asarray(list(set(lons)))
    stuff = np.empty( (lats.shape[0], lons.shape[0] ), dtype='object') 
    with open(file, 'r') as f: 
        for line in f: 
            i, j = line.split(' ')[:2]
            dct = ' '.join(line.strip().split(' ')[2:] ).replace("'", '"').replace('None', '"None"').replace('False', 'false').replace('True', 'true').replace('nan', '{}')
            stuff[list(lats).index(i), list(lons).index(j)] = json.loads( dct )  
    return xr.DataArray(stuff, name='params', dims=['latitude', 'longitude'], coords={'latitude': lats.astype('float'), 'longitude': lons.astype(float)}).sortby('latitude').sortby('longitude')


def save_parameters(params, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, dest='test.params'):
    #x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(params, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    with open(dest, 'w') as f:
        for i in params.coords['latitude'].values:
            for j in params.coords['longitude'].values:
                dct = {'latitude': i, 'longitude': j}
                f.write("{} {} {}\n".format(i, j, params.sel(**dct).values) ) 