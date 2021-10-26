from .base_classifier import *
from .classifiers import *


tercile_classifiers = [eLogisticRegression, eMultivariateLogisticRegression, eRandomForest, eNaiveBayes, eMultiLayerPerceptron, ePOELM]
arbitrary_classifiers = [cMultiLayerPerceptron, cNaiveBayes, cRandomForest, cPOELM]
