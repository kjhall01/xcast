import copy
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm

class ELRClassifier:
	def __init__(self, pca=-999, preprocessing='none', verbose=False, **kwargs):
		self.kwargs = kwargs
		self.an_thresh = 0.67
		self.bn_thresh = 0.33
		self.pca = pca
		self.preprocessing = preprocessing
		self.verbose=verbose

	def fit(self, x, y):
		self.models = [MultivariateELRClassifier(pca=self.pca, preprocessing=self.preprocessing, verbose=self.verbose, **self.kwargs) for i in range(x.shape[1])]
		for i in range(x.shape[1]):
			self.models[i].bn_thresh = self.bn_thresh
			self.models[i].an_thresh = self.an_thresh
			self.models[i].fit(x[:,i].reshape(-1,1), y)

	def predict_proba(self, x):
		res = []
		for i in range(x.shape[1]):
			res.append(self.models[i].predict_proba(x[:,i].reshape(-1,1)))
		res = np.stack(res, axis=0)
		return np.nanmean(res, axis=0)

	def predict(self, x):
		res = []
		for i in range(x.shape[1]):
			res.append(self.models[i].predict_proba(x[:,i].reshape(-1,1)))
		res = np.stack(res, axis=0)
		return np.argmax(np.nanmean(res, axis=0), axis=-1)




class MultivariateELRClassifier:
	def __init__(self, pca=-999, preprocessing='none', verbose=False, **kwargs):
		self.kwargs = kwargs
		self.pca_retained = pca if pca != -1 else None
		self.an_thresh = 0.67
		self.bn_thresh = 0.33
		self.preprocessing = preprocessing
		self.verbose=verbose

	def fit(self, x, y):
		# first, take care of preprocessing
		x2 = copy.deepcopy(x)
		y2 = copy.deepcopy(y)
		if self.preprocessing == 'std':
			self.mean, self.std = x.mean(axis=0), x.std(axis=0)
			x2 = (copy.deepcopy(x) - self.mean) / self.std # scales to std normal dist
			if self.verbose:
				print('{} Applied Standard Normal Scaling to ELR '.format(dt.datetime.now()))
		if self.preprocessing == 'minmax':
			self.min, self.max = x.min(axis=0), x.max(axis=0)
			x2 = ((copy.deepcopy(x) - self.min) / (self.max - self.min)  ) * 2 - 1 #scales to [-1, 1]
			if self.verbose:
				print('{} Applied MinMax Scaling to ELR '.format(dt.datetime.now()))

		# now, if anything needs to train a PCA on the input data, do it (PCA initialization, or PCA transformation)
		if self.pca_retained != -999:
			self.pca = PCA(n_components=None if self.pca_retained == -1 else self.pca_retained )
			self.pca.fit(x2)
			if self.verbose:
				print('{} Fit PCA on X with n_compnents={} for ELR'.format(dt.datetime.now(), None if self.pca_retained==-999 or self.pca_retained == -1 else self.pca_retained))
			x2 = self.pca.transform(x2)
			if self.verbose:
				print('{} Applied PCA Transformation to X '.format(dt.datetime.now()))

		try:
			y2 = np.vstack([np.sum(y2[:,1:], axis=-1).reshape(-1,1), y2[:,2].reshape(-1,1)]) # exceeded 0.33, exceeded 0.66
			x_bn = np.hstack([x2, np.ones((x2.shape[0], 1)) * self.bn_thresh])
			x_an = np.hstack([x2, np.ones((x2.shape[0], 1)) * self.an_thresh])
			x2 = np.vstack([x_bn, x_an])
			#x = sm.add_constant(x)
			model = sm.GLM(y2, sm.add_constant(x2, has_constant='add'), family=sm.families.Binomial())
			self.model = model.fit()
			if self.verbose:
				print('{} Fit ELR '.format(dt.datetime.now()))
		except:
			pass


	def predict_proba(self, x):
		x2 = copy.deepcopy(x)
		if self.preprocessing == 'std':
			x2 = (copy.deepcopy(x) - self.mean) / self.std # scales to std normal dist
			if self.verbose:
				print('{} Applied Standard Normal Scaling to ELR '.format(dt.datetime.now()))
		if self.preprocessing == 'minmax':
			x2 = ((copy.deepcopy(x) - self.min) / (self.max - self.min)  ) * 2 - 1 #scales to [-1, 1]
			if self.verbose:
				print('{} Applied MinMax Scaling to ELR '.format(dt.datetime.now()))

		# now, if anything needs to train a PCA on the input data, do it (PCA initialization, or PCA transformation)
		if self.pca_retained != -999:
			x2 = self.pca.transform(x2)
			if self.verbose:
				print('{} Applied PCA Transformation to X '.format(dt.datetime.now()))

		try:
			x_bn = np.hstack([x2, np.ones((x.shape[0], 1)) * self.bn_thresh])
			x_bn = sm.add_constant(x_bn, has_constant='add')
			bn = 1 - self.model.predict(x_bn).reshape(-1, 1)
			x_an = np.hstack([x2, np.ones((x.shape[0], 1)) * self.an_thresh])
			x_an = sm.add_constant(x_an, has_constant='add')
			an = self.model.predict(x_an).reshape(-1, 1)
			nn = 1 - (an + bn)
			return np.hstack([bn, nn, an])
		except:
			return np.hstack([np.ones((x2.shape[0], 1)) * 0.33, np.ones((x2.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])


	def predict(self, x):
		x2 = copy.deepcopy(x)
		if self.preprocessing == 'std':
			x2 = (copy.deepcopy(x) - self.mean) / self.std # scales to std normal dist
			if self.verbose:
				print('{} Applied Standard Normal Scaling to ELR '.format(dt.datetime.now()))
		if self.preprocessing == 'minmax':
			x2 = ((copy.deepcopy(x) - self.min) / (self.max - self.min)  ) * 2 - 1 #scales to [-1, 1]
			if self.verbose:
				print('{} Applied MinMax Scaling to ELR '.format(dt.datetime.now()))

		# now, if anything needs to train a PCA on the input data, do it (PCA initialization, or PCA transformation)
		if self.pca_retained != -999:
			x2 = self.pca.transform(x2)
			if self.verbose:
				print('{} Applied PCA Transformation to X '.format(dt.datetime.now()))

		try:
			x_bn = np.hstack([x2, np.ones((x.shape[0], 1)) * self.bn_thresh])
			x_bn = sm.add_constant(x_bn, has_constant='add')
			bn = 1 - self.model.predict(x_bn).reshape(-1, 1)
			x_an = np.hstack([x2, np.ones((x.shape[0], 1)) * self.an_thresh])
			x_an = sm.add_constant(x_an, has_constant='add')
			an = self.model.predict(x_an).reshape(-1, 1)
			nn = 1 - (an + bn)
			return np.argmax(np.hstack([bn, nn, an]), axis=-1)
		except:
			return np.argmax(np.hstack([np.ones((x2.shape[0], 1)) * 0.33, np.ones((x2.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33]), axis=-1)
