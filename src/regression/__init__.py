from .base_regressor import *
from .regressors import *

all_regressors = [EnsembleMean, BiasCorrectedEnsembleMean, rMultipleLinearRegression, rPoissonRegression, rGammaRegression, rMultiLayerPerceptron, rRandomForest, rRidgeRegression, rExtremeLearningMachine]
