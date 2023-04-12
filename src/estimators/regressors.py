import numpy as np

from ..flat_estimators.mlr import linear_regression
from ..flat_estimators.einstein_elm import  extreme_learning_machine
from ..flat_estimators.einstein_epoelm import  epoelm
from ..flat_estimators.qrf import quantile_regression_forest
from .base_estimator import BaseEstimator


class MLR(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = linear_regression


class ELM(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = extreme_learning_machine

class QRF(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = quantile_regression_forest


class EPOELM(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = epoelm

