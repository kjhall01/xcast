import numpy as np
from ..flat_estimators.classifiers import *
from ..classification.base_classifier import *
from ..preprocessing import *

class MemberCountMME:
	def __init__(self, **kwargs):
		self.use_dask=use_dask

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33,  explicit=False, parallel_in_memory=True ):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		transformed = []
		for i in range(X1.shape[list(X1.dims).index(x_feature_dim)]):
			dc = {x_feature_dim: i}
			to_transform = X1.isel(**dc)
			to_transform = to_transform.expand_dims(x_feature_dim)
			to_transform.coords[x_feature_dim] = [i]
			transformed.append(self.onehot.transform(to_transform, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim))
		X1 = xr.concat(transformed, x_feature_dim)

		return X1.sum(x_feature_dim) / X1.sum(x_feature_dim).sum('C')


class StandardMemberCountMME:
	def __init__(self, **kwargs):
		self.use_dask=use_dask

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33, explicit=False, parallel_in_memory=True  ):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.normx = Normal()
		self.normx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.normy = Normal()
		self.normy.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		X1 = self.normx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		transformed = []
		for i in range(X1.shape[list(X1.dims).index(x_feature_dim)]):
			dc = {x_feature_dim: i}
			to_transform = X1.isel(**dc)
			to_transform = to_transform.expand_dims(x_feature_dim)
			to_transform.coords[x_feature_dim] = [i]
			to_transform = self.normy.inverse_transform(to_transform, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
			transformed.append(self.onehot.transform(to_transform, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim))
		X1 = xr.concat(transformed, x_feature_dim)

		return X1.sum(x_feature_dim) / X1.sum(x_feature_dim).sum('C')


class ElrMME(BaseClassifier):
	def __init__(self, an_thresh=0.67, bn_thresh=0.33, explicit=False, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = ELRClassifier
		self.kwargs['an_thresh'] = an_thresh
		self.kwargs['bn_thresh'] = bn_thresh

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33, explicit=False, parallel_in_memory=True  ):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.mmx = Normal()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose).rename({x_feature_dim:'C'})


class MultivariateElrMME(BaseClassifier):
	def __init__(self, an_thresh=0.67, bn_thresh=0.33, explicit=False, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = MultivariateELRClassifier
		self.kwargs['an_thresh'] = an_thresh
		self.kwargs['bn_thresh'] = bn_thresh

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33, explicit=False, parallel_in_memory=True  ):
		X1 = X.sel() #fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.mmx = Normal()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose).rename({x_feature_dim:'C'})

class ProbabilisticRandomForestMME(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = MultiClassRandomForestClassifier

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33, explicit=False, parallel_in_memory=True  ):
		X1 = X.sel()#fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose).rename({x_feature_dim:'C'})


class ProbabilisticMlpMME(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = MultiClassMLPClassifier

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33, explicit=False, parallel_in_memory=True  ):
		X1 = X.sel()#fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel()#fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose).rename({x_feature_dim:'C'})


class PoelmPcaMME(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = MultiClassPOELMClassifier

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33, explicit=False, n_components=2, parallel_in_memory=True ):
		X1 = X.sel()#fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel() #fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.pca = PrincipalComponentsAnalysis(n_components=n_components)
		self.pca.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.pca.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose).rename({x_feature_dim:'C'})



class PoelmMME(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = MultiClassPOELMClassifier

	def fit(self, X, Y, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', y_lat_dim='Y', y_lon_dim='X', y_sample_dim='T', y_feature_dim='M', lat_chunks=1, lon_chunks=1, feat_chunks=1, samp_chunks=1, verbose=False, an_thresh=0.67, bn_thresh=0.33, explicit=False, parallel_in_memory=True ):
		X1 = X.sel()#fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		Y1 = Y.sel()#fill_space_mean(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		self.regrid_coords_lat = Y1.coords[y_lat_dim].values
		self.regrid_coords_lon = Y1.coords[y_lon_dim].values
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		self.mmx = MinMax()
		self.mmx.fit(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

		self.onehot = RankedTerciles(low_thresh=bn_thresh, high_thresh=an_thresh, explicit=explicit)
		self.onehot.fit(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		Y1 = self.onehot.transform(Y1, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
		y_feature_dim = 'C'

		super().fit(X1, Y1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim,  lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks, verbose=verbose)

	def predict(self, X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', lat_chunks=1, lon_chunks=1 , feat_chunks=1, samp_chunks=1, verbose=False):
		X1 = fill_space_mean(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		#the following ignores the X1 part, just uses coords, but it actually will use previously saved if override=False, like the default is.
		if len(self.regrid_coords_lat)*len(self.regrid_coords_lon) > 1:
			X1 = regrid(X1, self.regrid_coords_lon, self.regrid_coords_lat, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, use_dask=self.use_dask, feat_chunks=feat_chunks, samp_chunks=samp_chunks)

		X1 = self.mmx.transform(X1, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
		return  super().predict(X1, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , lat_chunks=lat_chunks, lon_chunks=lon_chunks, feat_chunks=feat_chunks, samp_chunks=samp_chunks,verbose=verbose).rename({x_feature_dim:'C'})
