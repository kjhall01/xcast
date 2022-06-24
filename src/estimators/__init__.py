from .base_estimator import BaseEstimator
from .classifiers import cMemberCount, cMultivariateLogisticRegression, cExtendedLogisticRegression, cMultiLayerPerceptron, cNaiveBayes, cRandomForest, cPOELM
from .regressors import EnsembleMean, BiasCorrectedEnsembleMean, rMultipleLinearRegression, rPoissonRegression, rGammaRegression, rMultiLayerPerceptron, rRandomForest, rRidgeRegression, rExtremeLearningMachine
from .base_multioutput import MultiOutputRegressor

classifiers = [cMemberCount, cMultivariateLogisticRegression,
               cExtendedLogisticRegression, cMultiLayerPerceptron, cNaiveBayes, cRandomForest, cPOELM]
regressors = [EnsembleMean, BiasCorrectedEnsembleMean, rMultipleLinearRegression, rPoissonRegression,
              rGammaRegression, rMultiLayerPerceptron, rRandomForest, rRidgeRegression, rExtremeLearningMachine]
