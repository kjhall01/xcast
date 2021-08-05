from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression, GammaRegressor, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xarray as xr
import numpy as np
import dask.array as da
import uuid
import h5py
from ..core.utilities import *
from ..core.progressbar import *
from ..preprocessing.normal import *
from ..preprocessing.decomposition import *
from ..preprocessing.onehot import *
from .wrappers import *
from ..preprocessing.spatial import *

class BaseSpatialMME:
	def __init__(self, use_dask=False, **kwargs):
		self.use_dask=use_dask
		self.kwargs = kwargs
		self.models = None
		self.model_type = LinearRegression

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		self.normx = Normal()
		self.normx.fit(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.normx.transform(X.isel(), x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		X1 = X1.transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)
		self.pca = SpatialPCA(use_dask=self.use_dask, **self.kwargs)
		self.pca.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, verbose=verbose)
		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, verbose=verbose)

		X1 = X1.transpose( x_sample_dim,  x_lat_dim, x_lon_dim, x_feature_dim)
		x_data = X1.values.reshape(X1.shape[list(X1.dims).index(x_sample_dim)]*X1.shape[list(X1.dims).index(x_lat_dim)]*X1.shape[list(X1.dims).index(x_lon_dim)], X1.shape[list(X1.dims).index(x_feature_dim)]  )

		Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_data = Y1.values.reshape(Y1.shape[list(Y1.dims).index(y_lat_dim)]*Y1.shape[list(Y1.dims).index(y_lon_dim)]*Y1.shape[list(Y1.dims).index(y_sample_dim)], Y1.shape[list(Y1.dims).index(y_feature_dim)])

		self.total = Y1.shape[list(Y1.dims).index(y_lat_dim)]* Y1.shape[list(Y1.dims).index(y_lon_dim)]
		self.count = 0
		if verbose:
			self.prog = ProgressBar(self.total, label='Fitting SPCR:', step=1)
			self.prog.show(self.count)

		self.y_coords = Y1.coords
		self.model = self.model_type(**self.kwargs)
		self.model.fit(x_data, y_data)
		#self.models = []
		#for i in range(Y1.shape[list(Y1.dims).index(y_lat_dim)]):
		#	self.models.append([])
		#	for j in range(Y1.shape[list(Y1.dims).index(y_lon_dim)]):
		#		self.models[i].append(self.model_type())
		#		self.models[i][j].fit(x_data, y_data[i, j, :, :])
		#		self.count += 1
		#		if verbose:
		#			self.prog.show(self.count)
		if verbose:
			self.prog.finish()
			self.count = 0


	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.normx.transform(X.isel(), x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim, verbose=verbose)
		X1 = X1.transpose( x_sample_dim, x_lat_dim, x_lon_dim,  x_feature_dim,)

		x_data = X1.values.reshape(X1.shape[list(X1.dims).index(x_sample_dim)]*X1.shape[list(X1.dims).index(x_lat_dim)]*X1.shape[list(X1.dims).index(x_lon_dim)], X1.shape[list(X1.dims).index(x_feature_dim)]  )
		if verbose:
			self.prog = ProgressBar(self.total, label='Predicting SPCR:', step=1)
			self.prog.show(self.count)

		#results = []
		#for i in range(len(self.models)):
		#	results.append([])
		#	for j in range(len(self.models[0])):
		#		results[i].append(self.models[i][j].predict(x_data))
		#		self.count += 1
		#		if verbose:
		#			self.prog.show(self.count)


		results = self.model.predict(x_data).reshape(X1.shape[list(X1.dims).index(x_sample_dim)], X1.shape[list(X1.dims).index(x_lon_dim)], X1.shape[list(X1.dims).index(x_lat_dim)], len(self.y_coords[self.feature_dim].values) ) #np.asarray(results)
		if verbose:
			self.prog.finish()
			self.count = 0
		coords = {
			x_lat_dim: self.y_coords[self.lat_dim].values,
			x_lon_dim: self.y_coords[self.lon_dim].values,
			x_sample_dim: X1.coords[x_sample_dim].values,
			x_feature_dim: [i for i in range(results.shape[-1])]
		}
		dims = [x_sample_dim, x_lat_dim, x_lon_dim, x_feature_dim]
		return xr.DataArray(data=results, coords=coords, dims=dims)




class ProbabilisticSpatialMLP(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = MultiLayerPerceptronProbabilistic

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)


		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh)
		self.onehot.fit(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y1, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		return super().predict(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose ).rename({x_feature_dim:'C'})

class SpatialPOELM(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = POELM

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)


		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh)
		self.onehot.fit(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y1, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		return super().predict(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose ).rename({x_feature_dim:'C'})


class ProbabilisticSpatialRF(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = RandomForestProbabilistic

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)


		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh)
		self.onehot.fit(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y1, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		return super().predict(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose ).rename({x_feature_dim:'C'})

class SpatialELR(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = MultipleELR

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)


		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh)
		self.onehot.fit(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y1, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		return super().predict(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose ).rename({x_feature_dim:'C'})


class SpatialMultiELR(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = ELR

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)


		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh)
		self.onehot.fit(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y1, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		return super().predict(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose ).rename({x_feature_dim:'C'})



class SpatialMLP(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = MLPRegressor

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim

		self.normy = Normal()
		self.normy.fit(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		X1 = self.normy.transform(Y.isel(), y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		super().fit(X, Y, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		ret = super().predict(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose )
		return self.normy.inverse_transform(ret, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)

class SpatialPCR(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)


class SpatialELM(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = ELM

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)


class SpatialRidge(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = Ridge

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)


class SpatialRF(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = RandomForestRegressor

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

		self.normy = Normal()
		self.normy.fit(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		X1 = self.normy.transform(Y.isel(), y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim
		super().fit(X, Y, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M',  lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1, verbose=False ):
		ret = super().predict(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks,  feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose )
		return self.normy.inverse_transform(ret, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)


class SpatialGammaRegression(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = GammaRegressionOne

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim

		X1 = X.where(X > 0.00000000001, other=0.00000000001)
		Y1 = Y.where(Y > 0.00000000001, other=0.00000000001)

		super().fit(X, Y1, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)

class SpatialPoissonRegression(BaseSpatialMME):
	def __init__(self, use_dask=False, **kwargs):
		super().__init__( use_dask=use_dask, **kwargs)
		self.model_type = PoissonRegressionOne

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1,  feat_chunks=1, samp_chunks=1,an_thresh=None, bn_thresh=None, verbose=False ):
		check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.lat_dim , self.lon_dim = y_lat_dim, y_lon_dim
		self.sample_dim, self.feature_dim = y_sample_dim, y_feature_dim

		X1 = X.where(X > 0.00000000001, other=0.00000000001)
		Y1 = Y.where(Y > 0.00000000001, other=0.00000000001)
		super().fit(X, Y1, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_feature_dim=x_feature_dim, x_sample_dim=x_sample_dim,  y_lat_dim=y_lat_dim, y_lon_dim=y_lon_dim, y_feature_dim=y_feature_dim, y_sample_dim=y_sample_dim, lat_chunks=lat_chunks, lon_chunks=lon_chunks , feat_chunks=feat_chunks, samp_chunks=samp_chunks, an_thresh=an_thresh, bn_thresh=bn_thresh, verbose=verbose)
