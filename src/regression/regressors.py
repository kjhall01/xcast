import numpy as np
from sklearn.linear_model import LinearRegression, GammaRegressor, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from ..flat_estimators.regressors import *
from .base_regressor import *
from ..preprocessing import *
#import extremelearning as elm


class MultipleLinearRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = LinearRegression

class PoissonRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = PoissonRegressionOne

class GammaRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = GammaRegressionOne

class MultiLayerPerceptronRegression(BaseRegressor):
	def __init__(self, hidden_layer_sizes=None, **kwargs):
		if hidden_layer_sizes is not None:
			kwargs['hidden_layer_sizes'] = hidden_layer_sizes
		else:
			kwargs['hidden_layer_sizes'] = (5,)
		super().__init__(**kwargs)
		self.model_type = MLPRegressor

class RandomForestRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = RandomForestRegressor


class RidgeRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = Ridge

class ExtremeLearningMachineRegression(BaseRegressor):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.model_type = ELMRegressor
