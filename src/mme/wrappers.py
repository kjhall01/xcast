import hpelm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import GammaRegressor, PoissonRegressor
import statsmodels.api as sm
from sklearn.decomposition import PCA

class GammaRegressionOne:
	def __init__(self, **kwargs):
		self.model = GammaRegressor(**kwargs)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		return self.model.predict(x).reshape(-1,1)

class PoissonRegressionOne:
	def __init__(self, **kwargs):
		self.model = PoissonRegressor(**kwargs)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		return self.model.predict(x).reshape(-1,1)


class ELR:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.an_thresh = 0.67
		self.bn_thresh = 0.33

	def fit(self, x, y):
		try:
			y = np.vstack([np.sum(y[:,1:], axis=-1).reshape(-1,1), y[:,2].reshape(-1,1)]) # exceeded 0.33, exceeded 0.66
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			x = np.vstack([x_bn, x_an])
			#x = sm.add_constant(x)
			model = sm.GLM(y, sm.add_constant(x, has_constant='add'), family=sm.families.Binomial())
			self.model = model.fit()
		except:
			pass


	def predict(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			x_bn = sm.add_constant(x_bn, has_constant='add')
			bn = 1 - self.model.predict(x_bn).reshape(-1, 1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			x_an = sm.add_constant(x_an, has_constant='add')
			an = self.model.predict(x_an).reshape(-1, 1)
			nn = 1 - (an + bn)
			return np.hstack([bn, nn, an])
		except:
			return np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])


class MultipleELR:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.an_thresh = 0.67
		self.bn_thresh = 0.33

	def fit(self, x, y):
		self.models = [ELR(**self.kwargs) for i in range(x.shape[1])]
		for i in range(x.shape[1]):
			self.models[i].bn_thresh = self.bn_thresh
			self.models[i].an_thresh = self.an_thresh
			self.models[i].fit(x[:,i].reshape(-1,1), y)

	def predict(self, x):
		res = []
		for i in range(x.shape[1]):
			res.append(self.models[i].predict(x[:,i].reshape(-1,1)))
		res = np.stack(res, axis=0)
		return np.nanmean(res, axis=0)



class MultiLayerPerceptronProbabilistic:
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		self.model = MLPClassifier(**kwargs)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		preds = self.model.predict_proba(x)
		ret = []
		sums = np.sum(preds, axis=-1).reshape(-1,1)
		sums[sums < 1 ] = 1
		for i in range(preds.shape[0]):
			if np.sum(preds[i,:], axis=-1) >= 1:
				ret.append(preds[i,:] / sums[i,0])
			else:
				ret.append(softmax(preds[i,:], axis=-1))
		return  np.asarray(ret)

class NaiveBayesProbabilistic:
	def __init__(self, **kwargs):
		self.model = MultinomialNB(**kwargs)

	def fit(self, x, y):
		self.model.partial_fit(x, np.argmax(y, axis=-1), classes=[0,1,2])

	def predict(self, x):
		n = self.model.predict_proba(x)
		return n

class RandomForestProbabilistic:
	def __init__(self, **kwargs):
		self.model = MLPClassifier(**kwargs)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		return self.model.predict_proba(x)


class ELM:
	def __init__(self, hidden_layer_size=5, activation = 'sigm', **kwargs):
		self.hidden_layer_size = 5
		self.activation = activation

	def fit(self, x, y):
		self.model = hpelm.ELM(x.shape[1], y.shape[1])
		self.model.add_neurons(self.hidden_layer_size, self.activation)
		self.model.train(x, y, 'r')

	def predict(self, x):
		return self.model.predict(x)


class POELM:
	def __init__(self, hidden_layer_size=5, n_components=-1, **kwargs):
		self.hidden_layer_size = hidden_layer_size
		self.n_components = n_components

	def fit(self, x, y, c=1):
		y[y<0.5] = 0.0001
		y[y>0.5] = 0.9999
		assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		x_features, y_features = x.shape[1], y.shape[1]
		self.hidden_neurons = [ (np.random.randn(x_features), np.random.randn(1)) for i in range(self.hidden_layer_size)]
		self.H = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		if self.n_components is None or self.n_components >= 0:
			self.pca = PCA(n_components=self.n_components)
			self.H = self.pca.fit_transform(self.H)
		hth = np.dot(np.transpose(self.H), self.H)
		inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / c )
		ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))
		self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)

	def predict(self, x):
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		if self.n_components is None or self.n_components >= 0:
			h = self.pca.transform(h)
		ret = 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))
		sums =  np.sum(ret, axis=1)
		ret1 = ret / sums.reshape(-1,1)
		ret2 = softmax(ret, axis=-1)
		retfinal = np.ones(ret.shape)
		retfinal[sums >=1, :] = ret1[sums>=1, :]
		retfinal[sums < 1, :] = ret2[sums<1, :]
		return retfinal

	def _activate(self, a, x, b):
		return 1.0 / (1 + np.exp(-1* np.dot(a, x.T) + b) )

class ExtendedMultiLayerPerceptronProbabilistic:
	def __init__(self, **kwargs):
		self.model = MultiLayerPerceptronProbabilistic(**kwargs)
		self.an_thresh = 0.67
		self.bn_thresh = 0.33

	def fit(self, x, y):
		try:
			y = np.vstack([np.sum(y[:,1:], axis=-1).reshape(-1,1), y[:,2].reshape(-1,1)]) # exceeded 0.33, exceeded 0.66
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			x = np.vstack([x_bn, x_an])
			#x = sm.add_constant(x)
			self.model.fit(x, y)
		except:
			pass


	def predict(self, x):
		#try:
		x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
		bn = self.model.predict(x_bn)[:,0].reshape(-1,1)
		x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
		an = self.model.predict(x_an)[:,1].reshape(-1,1)
		nn = 1 - (an + bn)
		return np.hstack([bn, nn, an])

class ExtendedNB:
	def __init__(self, **kwargs):
		self.model = MultinomialNB(**kwargs)
		self.an_thresh = 0.67
		self.bn_thresh = 0.33

	def fit(self, x, y):
		try:
			y = np.vstack([np.sum(y[:,1:], axis=-1).reshape(-1,1), y[:,2].reshape(-1,1)]) # exceeded 0.33, exceeded 0.66
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			x = np.vstack([x_bn, x_an])
			#x = sm.add_constant(x)
			self.model.partial_fit(x, np.argmax(y, axis=-1), classes=[0,1])
		except:
			pass


	def predict(self, x):
		#try:
		x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
		bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
		x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
		an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
		nn = 1 - (an + bn)
		return np.hstack([bn, nn, an])
		#except:
		#	return np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])


class ExtendedPOELM:
	def __init__(self, hidden_layer_size=5, n_components=-1, **kwargs):
		self.hidden_layer_size = hidden_layer_size
		self.n_components = n_components
		self.an_thresh = 0.67
		self.bn_thresh = 0.33

	def fit(self, x, y, c=1):
		assert len(x.shape) == 2 and len(y.shape) ==2, 'wrong shape inputs for fit'
		y = np.vstack([1 - y[:,0].reshape(-1,1), y[:,2].reshape(-1,1)]) # exceeded 0.33, exceeded 0.66
		x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
		x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
		x = np.vstack([x_bn, x_an])
		y[y<0.5] = 0.0001
		y[y>0.5] = 0.9999

		x_features, y_features = x.shape[1], y.shape[1]
		self.hidden_neurons = [ (np.random.randn(x_features), np.random.randn(1)) for i in range(self.hidden_layer_size)]
		self.H = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		if self.n_components is None or self.n_components >= 0:
			self.pca = PCA(n_components=self.n_components)
			self.H = self.pca.fit_transform(self.H)
		hth = np.dot(np.transpose(self.H), self.H)
		inv_hth_plus_ic = np.linalg.pinv( hth + np.eye(hth.shape[0]) / c )
		ht_logs = np.dot(np.transpose(self.H), np.log(1 - y) - np.log(y))
		self.beta = -1 * np.dot( inv_hth_plus_ic, ht_logs)

	def predict(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = 1 - self._predict(x_bn).reshape(-1, 1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self._predict(x_an).reshape(-1, 1)
			nn = 1 - (an + bn)
			return np.hstack([bn, nn, an])

		except:
			return np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])

	def _predict(self, x):
		h = np.asarray([ self._activate(neuron[0], x, neuron[1]) for neuron in self.hidden_neurons]).T
		if self.n_components is None or self.n_components >= 0:
			h = self.pca.transform(h)
		return 1.0 / ( 1 + np.exp(-1* np.dot(h, self.beta)))


	def _activate(self, a, x, b):
		return 1.0 / (1 + np.exp(-1* np.dot(a, x.T) + b) )
