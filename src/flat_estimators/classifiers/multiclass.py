from .poelm import POELMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.special import softmax

class MultiClassPOELM(POELMClassifier):
	def predict_proba(self, x):
		ret = self.predict_proba(x)
		sums =  np.sum(ret, axis=1)
		ret1 = ret / sums.reshape(-1,1)
		ret2 = softmax(ret, axis=-1)
		retfinal = np.ones(ret.shape)
		retfinal[sums >=1, :] = ret1[sums>=1, :]
		retfinal[sums < 1, :] = ret2[sums<1, :]
		return retfinal


class MultiClassMLP:
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		self.model = MLPClassifier(**kwargs)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict_proba(self, x):
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


class MultiClassNaiveBayes:
	def __init__(self, **kwargs):
		self.model = MultinomialNB(**kwargs)

	def fit(self, x, y):
		self.model.partial_fit(x, np.argmax(y, axis=-1), classes=[0,1,2])

	def predict_proba(self, x):
		return self.model.predict_proba(x)

class MultiClassRandomForest:
	def __init__(self, **kwargs):
		self.model = RandomForestClassifier(**kwargs)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict_proba(self, x):
		return self.model.predict_proba(x)
