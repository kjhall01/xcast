from .einstein_elm import ELM
from .einstein_poelm import POELM
from .einstein_epoelm import EPOELM
from .qrf import QRF

from ...estimators import BaseEstimator


class rEinsteinLearningMachine(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = ELM

class rQuantileRandomForest(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = QRF


class rEPOELM(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = EPOELM


class rPOELM(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = POELM
