from ..flat_estimators.classifiers import *
from .base_classifier import *

class ExtendedPOELMClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ExtendedPOELM

class ExtendedMLPClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ExtendedMLP

class ExtendedNaiveBayesClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ExtendedNaiveBayes

class ExtendedRandomForestClassifier(BaseClassifier):
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		super().__init__(**kwargs)
		self.model_type = ExtendedRandomForest

class MultivarELRClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = MultivariateELRClassifier


class ExtendedLogisticRegressionClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ELRClassifier

class MultiClassPOELMClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = MultiClassPOELM

class MultiClassMLPClassifier(BaseClassifier):
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		super().__init__(**kwargs)
		self.model_type = MultiClassMLP

class MultiClassNaiveBayesClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = MultiClassNaiveBayes

class MultiClassRandomForestClassifier(BaseClassifier):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = MultiClassRandomForest
