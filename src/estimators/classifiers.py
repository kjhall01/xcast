from ..flat_estimators.elr import multivariate_extended_logistic_regression, extended_logistic_regression 
from ..flat_estimators.wrappers import rf_classifier, naive_bayes_classifier
from sklearn.neural_network import MLPClassifier
from .base_estimator import BaseEstimator
from ..core.utilities import guess_coords, check_all


class MELR(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = multivariate_extended_logistic_regression


class ELR(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = extended_logistic_regression
