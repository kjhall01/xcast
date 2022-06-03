from .base_estimator import BaseEstimator
from .classifiers import cMemberCount, cMultivariateLogisticRegression, cExtendedLogisticRegression, cMultiLayerPerceptron, cNaiveBayes, cRandomForest, cPOELM
from .regressors import EnsembleMean, BiasCorrectedEnsembleMean, rMultipleLinearRegression, rPoissonRegression, rGammaRegression, rMultiLayerPerceptron, rRandomForest, rRidgeRegression, rExtremeLearningMachine
from .cca import sCCA
from .base_multioutput import BaseMultiOutputRegressor
from .prep import GammaTransformer, EmpiricalTransformer

classifiers = [cMemberCount, cMultivariateLogisticRegression,
               cExtendedLogisticRegression, cMultiLayerPerceptron, cNaiveBayes, cRandomForest, cPOELM]
regressors = [EnsembleMean, BiasCorrectedEnsembleMean, rMultipleLinearRegression, rPoissonRegression,
              rGammaRegression, rMultiLayerPerceptron, rRandomForest, rRidgeRegression, rExtremeLearningMachine]
