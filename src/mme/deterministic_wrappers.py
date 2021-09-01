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
