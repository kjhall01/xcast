from .regressors import *
from .classifiers import *
from .einstein_elm import EinsteinLearningMachine

flat_regressors = [ELMRegressor]
flat_classifiers = [ELRClassifier, POELMClassifier, MultivariateELRClassifier ]
