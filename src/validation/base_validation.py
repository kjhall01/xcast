import xarray as xr
import xskillscore as xs
from ..core.utilities import *
from ..core.progressbar import *
import numpy as np


def cross_validate( MME, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M',  window=3, verbose=0, ND=1, bn_thresh=None, an_thresh=None, explicit=False, lat_chunks=1, lon_chunks=1, **kwargs ):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	crossval_x = CrossValidator(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, window=window)
	crossval_y = CrossValidator(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim, window=window)

	prog = ProgressBar(crossval_x.windows, step=1, label=" CROSSVALIDATING {}: ".format(MME.__name__))
	count = 0
	if verbose:
		prog.show(count)

	x_train, x_test = crossval_x.get_next_window()
	y_train, y_test = crossval_y.get_next_window()

	prediction_means, prediction_stds = [], []
	kwargs['ND'] = ND
	while x_train is not None and y_train is not None and x_test is not None and y_test is not None:
		mme  = MME(**kwargs)
		mme.fit(x_train, y_train, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_sample_dim=y_sample_dim, y_feature_dim=y_feature_dim, an_thresh=an_thresh, bn_thresh=bn_thresh, lat_chunks=lat_chunks, lon_chunks=lon_chunks, explicit=explicit)
		pred_means = mme.predict(x_test, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks)
		pred_stds = mme.predict(x_test, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks, mode='std')

		count += 1
		if verbose:
			prog.show(count)

		#preds = xr.concat(preds_temp, 'N')
		prediction_means.append(pred_means)
		prediction_stds.append(pred_stds)

		x_train, x_test = crossval_x.get_next_window()
		y_train, y_test = crossval_y.get_next_window()

	prediction_means = xr.concat(prediction_means, x_sample_dim)
	prediction_stds = xr.concat(prediction_stds, x_sample_dim)
	if verbose:
		prog.finish()

	data_vars = {'hindcasts': prediction_means, 'nd_stddev': prediction_stds}
	return xr.Dataset(data_vars, coords=prediction_stds.coords)





class CrossValidator:
	def __init__(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', window=3 ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		self.window = window + 1 if window % 2 == 0 else window
		self.X = X.isel()
		self.lat_dim, self.lon_dim = x_lat_dim, x_lon_dim
		self.sample_dim, self.feature_dim = x_sample_dim, x_feature_dim
		self.t_size = len(self.X.coords[x_sample_dim].values)
		self.windows = int(self.t_size // self.window) + 1
		self.range_top = self.window
		self.range_bottom = 0
		self.done = False


	def get_next_window(self, verbose=False):
		if self.done:
			return None, None
		t_ndcs = xr.DataArray([j for j in range(self.t_size) if not j >= self.range_bottom or not j < self.range_top], dims=[self.sample_dim])
		sel_dict = {self.sample_dim: t_ndcs}
		train = self.X.isel(**sel_dict)
		t_ndcs_test = xr.DataArray([j for j in range(self.t_size) if  j >= self.range_bottom and j < self.range_top], dims=[self.sample_dim])

		test_sel_dict = {self.sample_dim: t_ndcs_test}
		test = self.X.isel(**test_sel_dict)
		self.range_bottom += self.window
		self.range_top += self.window
		if self.range_top > self.t_size:
			self.range_top = self.t_size
		if self.range_bottom >= self.t_size:
			self.done = True

		if self.sample_dim not in train.dims:
			train = train.expand_dims({self.sample_dim:t_ndcs})

		if self.sample_dim not in test.dims:
			test = test.expand_dims({self.sample_dim:t_ndcs_test})
		return train, test
