from ..flat_estimators.classifiers import *
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
from .base_classifier import *

class ePOELM(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ExtendedPOELMClassifier

class eMultiLayerPerceptron(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ExtendedMLPClassifier

class eNaiveBayes(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ExtendedNaiveBayesClassifier

class eRandomForest(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ExtendedRandomForestClassifier

class eMultivariateLogisticRegression(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = MultivariateELRClassifier


class eLogisticRegression(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ELRClassifier


class cMultiLayerPerceptron(BaseClassifier):
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		super().__init__(**kwargs)
		self.model_type = MLPClassifier

class cNaiveBayes(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = NaiveBayesClassifier

class cRandomForest(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = RFClassifier

class cPOELM(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = POELMClassifier
