from .poelm import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.special import softmax


class ExtendedPOELMClassifier:
	def __init__(self, **kwargs):
		self.model = POELMClassifier(**kwargs)
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

	def predict_proba(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.hstack([bn, nn, an])
		except:
			return np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])

	def predict(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.argmax(np.hstack([bn, nn, an]), axis=-1)
		except:
			return np.argmax(np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33]), axis=-1)



class ExtendedMLPClassifier:
	def __init__(self, **kwargs):
		self.model = MLPClassifier(**kwargs)
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

	def predict_proba(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.hstack([bn, nn, an])
		except:
			return np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])

	def predict(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.argmax(np.hstack([bn, nn, an]), axis=-1)
		except:
			return np.argmax(np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33]), axis=-1)


class ExtendedNaiveBayesClassifier:
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

	def predict_proba(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.hstack([bn, nn, an])
		except:
			return np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])

	def predict(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.argmax(np.hstack([bn, nn, an]), axis=-1)
		except:
			return np.argmax(np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33]), axis=-1)


class ExtendedRandomForestClassifier:
	def __init__(self, **kwargs):
		self.model = RandomForestClassifier(**kwargs)
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

	def predict_proba(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.hstack([bn, nn, an])
		except:
			return np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33])

	def predict(self, x):
		try:
			x_bn = np.hstack([x, np.ones((x.shape[0], 1)) * self.bn_thresh])
			bn = self.model.predict_proba(x_bn)[:,0].reshape(-1,1)
			x_an = np.hstack([x, np.ones((x.shape[0], 1)) * self.an_thresh])
			an = self.model.predict_proba(x_an)[:,1].reshape(-1,1)
			nn = 1 - (an + bn)
			return np.argmax(np.hstack([bn, nn, an]), axis=-1)
		except:
			return np.argmax(np.hstack([np.ones((x.shape[0], 1)) * 0.33, np.ones((x.shape[0], 1)) * 0.34, np.ones((x.shape[0], 1)) * 0.33]), axis=-1)
