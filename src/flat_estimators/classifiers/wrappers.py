import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from scipy.special import softmax

def insert_zeros(x, i):
	print(f'INSERTING ZEROS AT COLUMN {i}')
	cols = x.shape[1]
	if cols == 1:
		if i == 0:
			return np.hstack([np.zeros(x.shape), x])
		elif i == 1:
			return np.hstack([x, np.zeros(x.shape)])
		else:
			return np.hstack([x, np.zeros(x.shape)])
	else:
		if i == 0:
			return np.hstack([np.zeros((x.shape[0], 1)), x])
		elif i == cols:
			return np.hstack([x, np.zeros((x.shape[0], 1))])
		elif i == 1:
			if cols > 2:
				return np.hstack([x[:, 0].reshape(-1,1), np.zeros((x.shape[0],1)), x[:,1:] ])
			else:
				return np.hstack([x[:, 0].reshape(-1,1), np.zeros((x.shape[0],1)), x[:,1:].reshape(-1,1) ])
		elif i == cols-1:
			if cols > 2:
				return np.hstack([x[:, :cols-1], np.zeros((x.shape[0],1)), x[:,cols-1:].reshape(-1,1) ])
			else:
				return np.hstack([x[:, :cols-1].reshape(-1,1), np.zeros((x.shape[0],1)), x[:,cols-1:].reshape(-1,1) ])
		else:
			return np.hstack([x[:,:i], np.zeros((x.shape[0], 1)), x[:, i:]])

class RFClassifier:
	def __init__(self, **kwargs):
		self.model = RandomForestClassifier(**kwargs)

	def fit(self, x, y):
		self.model.fit(x, np.argmax(y, axis=1))
		self.classes = [i for i in range(y.shape[1])]

	def predict_proba(self, x):
		ret = self.model.predict_proba(x)
		for i in self.classes:
			if i not in self.model.classes_:
				ret = insert_zeros(ret, i)
		return ret

	def predict(self, x):
		ret = self.predict_proba(x)
		return np.argmax(ret, axis=1)

class NanClassifier:
	def __init__(self, **kwargs):
		self.model=None
		for key in kwargs.keys():
			setattr(self, key, kwargs[key])

	def fit(self, x, *args ):
		self.x_features = x.shape[1]
		if len(args) > 0: 
			y = args[0]
			self.y_features = y.shape[1]

	def transform(self, x):
		assert self.x_features == x.shape[1]
		ret = np.empty((x.shape[0], self.n_components))
		ret[:] = np.nan
		return ret

	def predict(self, x):
		assert self.x_features == x.shape[1]
		ret = np.empty((x.shape[0], 1))
		ret[:] = np.nan
		return ret

	def predict_proba(self, x):
		assert self.x_features == x.shape[1]
		ret = np.empty((x.shape[0], self.y_features))
		ret[:] = np.nan
		return ret

class NaiveBayesClassifier:
	def __init__(self, **kwargs):
		self.model = MultinomialNB(**kwargs)

	def fit(self, x, y):
		self.model.partial_fit(x, np.argmax(y, axis=-1), classes=[i for i in range(y.shape[1])])

	def predict_proba(self, x):
		return self.model.predict_proba(x)

	def predict(self, x):
		ret = self.model.predict_proba(x)
		return np.argmax(ret, axis=1)
