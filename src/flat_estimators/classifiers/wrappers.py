import numpy as np

class NanClassifier:
	def __init__(self, **kwargs):
		self.model=None

	def fit(self, x, y ):
		self.x_features = x.shape[1]
		self.y_features = y.shape[1]

	def transform(self, x):
		assert self.x_features == x.shape[1]
		ret = np.empty((x.shape[0], 1))
		ret[:] = np.nan
		return ret

	def predict(self, x):
		assert self.x_features == x.shape[1]
		ret = np.empty((x.shape[0], self.y_features))
		ret[:] = np.nan
		return ret
