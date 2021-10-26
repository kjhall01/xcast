from .regressors import *
from .classifiers import *

flat_regressors = [ELMRegressor]
flat_classifiers = [ELRClassifier, POELMClassifier, MultivariateELRClassifier, ExtendedPOELMClassifier, ExtendedMLPClassifier, ExtendedNaiveBayesClassifier, ExtendedRandomForestClassifier ]
